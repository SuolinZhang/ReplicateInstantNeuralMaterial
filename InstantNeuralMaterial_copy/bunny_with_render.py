############################################################################
# Bunny with Environment Map and Path Tracer - Working Script
# Usage: Mogwai --script bunny_with_render.py
############################################################################

# Load the bunny scene with environment map
m.loadScene("test_scenes/bunny.pyscene")

# Create a proper render graph for path tracing
def render_graph_BunnyPathTracer():
    g = RenderGraph("BunnyPathTracer")

    # Create passes
    PathTracer = createPass("PathTracer", {'samplesPerPixel': 1})
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16, 'useAlphaTest': True})
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})

    # Add passes to graph
    g.addPass(PathTracer, "PathTracer")
    g.addPass(VBufferRT, "VBufferRT")
    g.addPass(AccumulatePass, "AccumulatePass")
    g.addPass(ToneMapper, "ToneMapper")

    # Connect the passes with edges
    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("VBufferRT.mvec", "PathTracer.mvec")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    # Mark the final output
    g.markOutput("ToneMapper.dst")

    return g

# Create and add the render graph
BunnyPathTracer = render_graph_BunnyPathTracer()
m.addGraph(BunnyPathTracer)
m.setActiveGraph(BunnyPathTracer)

print("Bunny scene loaded with proper PathTracer render graph!")
print("Ready to render...")