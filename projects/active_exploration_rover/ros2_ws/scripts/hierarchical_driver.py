"""Persistent app-server destination transport. No shell/MCP/navigation tools.

The transport bounds externally submitted turn/start messages at 20. Internal
inference requests may exceed that count. No automatic application retries.
Model execution requires explicit opt-in; offline tests do not spawn Codex.
"""
import json
import queue
import subprocess
import tempfile
import threading
import time
import tomllib
from pathlib import Path
from collections import deque

PROMPT='''Select exactly one current approach ID from the onboard candidate list. Local robotics plans and follows the route. Use camera, lidar, online map and recent outcomes to explore unknown space. Local blockage is not global exhaustion: consider alternate approaches and previously observed free space. Candidate reachability is provisional and revalidated locally. Return concise JSON with destination_id and reason. Never invent coordinates, issue movement commands, or use outside tools. No later destination is committed.'''
SCHEMA={'type':'object','properties':{'destination_id':{'type':'string'},'reason':{'type':'string'}},'required':['destination_id','reason'],'additionalProperties':False}
INPUT_SPEC_VERSION='hierarchical-onboard-v2'
FRONTIER_SIZE_DEFINITION='frontier_component_cell_count is the number of cells in the frontier component. It is not free-space area, accessible-area coverage, or a guarantee of traversability.'
PROMPT += '\nInput specification: '+INPUT_SPEC_VERSION+'. The compatible candidate field "cells" means frontier_component_cell_count. '+FRONTIER_SIZE_DEFINITION


class PersistentDriver:
    def __init__(self,folder,*,enable_model=False):
        if not enable_model:
            raise RuntimeError('model_execution_disabled: explicit authorization/opt-in required')
        self.folder=folder;self.tmp=tempfile.TemporaryDirectory(prefix='rover-goal-driver-')
        self.log=(folder/'driver_rpc.jsonl').open('a',buffering=1)
        self.counter=0;self.turns=0;self.thread_id=None;self.turn_id=None;self.closed=False
        self.inbox=queue.Queue();self.deferred=deque();self.write_lock=threading.Lock()
        config={'model':'gpt-5.6-luna','model_reasoning_effort':'low','web_search':'disabled',
                'approval_policy':'never','sandbox_mode':'read-only','mcp_servers':{},
                **{'features.'+k:False for k in ['shell_tool','view_image','apps','plugins','multi_agent','memories','browser_use','computer_use','image_generation','skill_search']},
                'features.skip_host_skill_discovery':True}
        # Disable every installed MCP by explicit leaf override (empty tables
        # alone need not erase merged user configuration). Empty cwd excludes repo.
        user_config=Path.home()/'.codex/config.toml'
        if user_config.exists():
            for name in tomllib.loads(user_config.read_text()).get('mcp_servers',{}):
                config['mcp_servers.'+name+'.enabled']=False
        cmd=['codex','app-server','--stdio']
        for k,v in config.items():cmd+=['-c',k+'='+json.dumps(v)]
        (folder/'driver_configuration.json').write_text(json.dumps(dict(command=cmd,config=config,
            model='gpt-5.6-luna',effort='low',input_spec_version=INPUT_SPEC_VERSION,submitted_turn_cap=20,actual_inference_cap_enforceable=False),indent=2))
        self.stderr=(folder/'driver_stderr.log').open('w')
        self.process=subprocess.Popen(cmd,cwd=self.tmp.name,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,bufsize=1)
        self.reader=threading.Thread(target=self.read,daemon=True);self.reader.start()
        self.rpc('initialize',{'clientInfo':{'name':'rover_hierarchy','version':'1'},'capabilities':{'experimentalApi':False}})
        self.send({'method':'initialized','params':{}})
        result=self.rpc('thread/start',{'model':'gpt-5.6-luna','cwd':self.tmp.name,'approvalPolicy':'never','sandbox':'read-only','baseInstructions':PROMPT,'ephemeral':True,'config':config})
        if result['model']!='gpt-5.6-luna' or result.get('reasoningEffort')!='low' or result.get('instructionSources'):
            self.close();raise RuntimeError('unexpected_model_settings_or_instruction_sources')
        self.thread_id=result['thread']['id']
        (folder/'hierarchical_model_metadata.json').write_text(json.dumps(result,indent=2))

    def read(self):
        for line in self.process.stdout:
            try:event=json.loads(line)
            except ValueError:continue
            self.inbox.put(event)
        self.inbox.put({'fatal':'transport_closed'})

    def send(self,event):
        with self.write_lock:
            if self.closed:raise RuntimeError('driver_closed')
            if event.get('method')=='turn/start':
                if self.turns>=20: raise RuntimeError('decision_turn_limit')
                self.turns+=1  # Reserve before write; failed writes are not retried.
            self.log.write(json.dumps({'direction':'sent','wall_s':time.monotonic(),'message':event})+'\n')
            self.process.stdin.write(json.dumps(event)+'\n');self.process.stdin.flush()

    def receive(self,deadline,allow_deferred=True):
        if allow_deferred and self.deferred: return self.deferred.popleft()
        event=self.inbox.get(timeout=max(.001,deadline-time.monotonic()))
        self.log.write(json.dumps({'direction':'received','wall_s':time.monotonic(),'message':event})+'\n')
        if 'fatal' in event:raise RuntimeError(event['fatal'])
        if 'method' in event and 'id' in event:raise RuntimeError('unexpected_server_tool_or_approval_request')
        text=json.dumps(event).lower()
        if any(x in text for x in ['usage limit','insufficient_quota','quota exceeded']):raise RuntimeError('quota_exhaustion')
        return event

    def rpc(self,method,params):
        self.counter+=1;rid=self.counter;self.send({'id':rid,'method':method,'params':params})
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            e=self.receive(deadline,allow_deferred=False)
            if e.get('id')==rid:
                if 'error' in e:raise RuntimeError(str(e['error']))
                return e['result']
            self.deferred.append(e)
        raise RuntimeError('transport_timeout')

    def choose(self,payload):
        if self.closed or self.turns>=20:raise RuntimeError('decision_turn_limit')
        started=time.monotonic()
        inputs=[{'type':'text','text':json.dumps(payload['snapshot'])}]+[{'type':'image','url':im} for im in payload['images']]
        result=self.rpc('turn/start',{'threadId':self.thread_id,'model':'gpt-5.6-luna','effort':'low','input':inputs,'outputSchema':SCHEMA})
        self.turn_id=result['turn']['id'];answer=None;deadline=started+55
        while time.monotonic()<deadline:
            e=self.receive(deadline);p=e.get('params',{})
            if e.get('method')=='item/completed' and p.get('item',{}).get('type')=='agentMessage':answer=p['item']['text']
            if e.get('method')=='turn/completed' and p.get('turn',{}).get('id')==self.turn_id:
                self.turn_id=None
                if p['turn']['status']!='completed':raise RuntimeError('model_turn_failed:'+json.dumps(p['turn'].get('error')))
                value=json.loads(answer)
                return value['destination_id']
        self.cancel();raise RuntimeError('decision_timeout')

    def cancel(self):
        if not self.closed and self.thread_id and self.turn_id:
            self.counter+=1;self.send({'id':self.counter,'method':'turn/interrupt','params':{'threadId':self.thread_id,'turnId':self.turn_id}})

    def close(self):
        if self.closed:return
        self.cancel();self.closed=True
        self.process.terminate()
        try:self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.reader.join(1);self.stderr.close();self.log.close();self.tmp.cleanup()
