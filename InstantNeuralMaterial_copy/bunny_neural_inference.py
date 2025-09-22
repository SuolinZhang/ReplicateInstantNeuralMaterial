############################################################################
# Bunny with Neural Material Inference
# Usage: Mogwai --script bunny_neural_inference.py
############################################################################

import os
from falcor import *

def render_graph_NeuralMatRendering():
    g = RenderGraph("NeuralMatRendering")

    # Create NeuralMatRendering pass - this is a complete path tracer
    NeuralMat = createPass("NeuralMatRendering", {
        'maxBounces': 3,
        'samplesPerPixel': 16,
        'modelName': 'LEATHER11'  # Use the leather11 model we set up
    })

    # Create supporting passes for accumulation and tone mapping
    AccumulatePass = createPass("AccumulatePass", {
        'enabled': True,
        'precisionMode': 'Single'
    })

    ToneMapper = createPass("ToneMapper", {
        'autoExposure': False,
        'exposureCompensation': 0.0
    })

    # Add passes to graph
    g.addPass(NeuralMat, "NeuralMatRendering")
    g.addPass(AccumulatePass, "AccumulatePass")
    g.addPass(ToneMapper, "ToneMapper")

    # Connect the passes
    g.addEdge("NeuralMatRendering.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    # Mark the final output
    g.markOutput("ToneMapper.dst")

    return g

# Load the bunny scene with neural material ID
m.loadScene("bunny_envmap.pyscene")

# Create and add the neural material render graph
NeuralMatGraph = render_graph_NeuralMatRendering()
m.addGraph(NeuralMatGraph)
m.setActiveGraph(NeuralMatGraph)

print("Bunny scene loaded with Neural Material Rendering!")
print("Using LEATHER11 neural material model")
print("Ready for neural material inference...")

print("Press F12 to save the current frame as a PNG screenshot")
print("The screenshot will be saved in the screenshots/ directory")
