from falcor import *

def render_graph_HFTracing():
    g = RenderGraph("HFTracing")
    AccumulateSTPass = createPass("AccumulateSTPass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulateSTPass, "AccumulateSTPass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    HFTracing = createPass("HFTracing", {'maxBounces': 3})
    g.addPass(HFTracing, "HFTracing")
    g.addPass(createPass("OptixDenoiser", {}), "Denoiser")
    # VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    # g.addPass(VBufferRT, "VBufferRT")
    g.addEdge("AccumulateSTPass.output", "ToneMapper.src")
    # g.addEdge("VBufferRT.vbuffer", "HFTracing.vbuffer")
    # g.addEdge("VBufferRT.viewW", "HFTracing.viewW")
    g.addEdge("HFTracing.color", "AccumulateSTPass.input")
    g.addEdge("ToneMapper.dst", "Denoiser.color")
    g.markOutput("Denoiser.output")
    return g

HFTracing = render_graph_HFTracing()
try: m.addGraph(HFTracing)
except NameError: None
