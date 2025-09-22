############################################################################
# Bunny with Environment Map Example
# Usage: Mogwai --script bunny_envmap.py
############################################################################

import Falcor

# Create materials
floor = Material('Floor')
floor.baseColor = float4(0.7, 0.7, 0.7, 1.0)
floor.roughness = 0.1

bunny_mat = Material('BunnyMaterial')
bunny_mat.baseColor = float4(0.9, 0.6, 0.4, 1.0)
bunny_mat.roughness = 0.2
bunny_mat.metallic = 0.8

# Create geometry
bunnyMesh = TriangleMesh.createFromFile('meshes/bunny.obj')

# Create mesh instances
sceneBuilder.addMeshInstance(
    sceneBuilder.addTriangleMesh(bunnyMesh, bunny_mat),
    sceneBuilder.addNode('Bunny', Transform(scaling=float3(0.4), translation=float3(0, 0.1, 0)))
)

# Add environment map
sceneBuilder.envMap = EnvMap('envmaps/hallstatt4_hd.hdr')

# Create camera
camera = Camera()
camera.position = float3(0, 0.5, 1.5)
camera.target = float3(0, 0.3, 0)
camera.up = float3(0, 1, 0)
sceneBuilder.addCamera(camera)