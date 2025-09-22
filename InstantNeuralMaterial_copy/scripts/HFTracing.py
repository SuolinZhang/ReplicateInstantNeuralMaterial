from falcor import *

def render_graph_HFTracing():
    g = RenderGraph("HFTracing")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    HFTracing = createPass("HFTracing", {'maxBounces': 3})
    g.addPass(HFTracing, "HFTracing")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    g.addPass(VBufferRT, "VBufferRT")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    # g.addEdge("VBufferRT.vbuffer", "HFTracing.vbuffer")
    # g.addEdge("VBufferRT.viewW", "HFTracing.viewW")
    g.addEdge("HFTracing.color", "AccumulatePass.input")
    g.markOutput("ToneMapper.dst")
    return g

HFTracing = render_graph_HFTracing()
try: m.addGraph(HFTracing)
except NameError: None
