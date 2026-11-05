"""Single reserved service-room proposal, instantiated without outcome search."""
import math
import transfer_layouts


def spec():
    a=[-4,-2,math.atan2(1,2)];b=[-4,2,-math.atan2(1,2)]
    c=[4,2,math.atan2(-1,-2)];d=[4,-2,math.atan2(1,-2)]
    dock=[-4,-3.5,0]
    return dict(bounds=[-6,6,-5,5],
        walls=[[-6,0,.12,10.12],[6,0,.12,10.12],[0,-5,12.12,.12],[0,5,12.12,.12],[0,0,4,3]],
        furniture=[[-5.25,.5,.6,1],[5.2,-.8,.7,1.2]],
        rearranged=[[-5.15,2,.6,1],[5.15,1.1,.7,1.2]],additions=[],
        start=dock,fixtures=[a,b,c,d],names=['west lower fitting','west upper fitting','east upper fitting','east lower fitting'],
        path=[(a,0),(b,1),([-4,3,0],None),([4,3,0],None),(c,2),(d,3),([4,-3,0],None),([-4,-3,0],None),(b,4),(a,5),(dock,None)],
        workflow='Service room with fixed central work island; two perimeter carts relocate while physical inspection fittings and access routes remain fixed.')


def make(out,condition,remedy):
    original=transfer_layouts._spec
    transfer_layouts._spec=lambda name:spec() if name=='service_island' else original(name)
    try:return transfer_layouts.make(out,'service_island',condition,remedy)
    finally:transfer_layouts._spec=original
