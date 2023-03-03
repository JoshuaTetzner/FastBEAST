using CompScienceMeshes

##
src = CompScienceMeshes.meshrectangle(1.0, 1.0, 0.4, 3)
src2 = CompScienceMeshes.rotate(src, SVector(0, pi/2, 0))
src2 = CompScienceMeshes.rotate(src2, SVector(0, 0, pi))
src2 = CompScienceMeshes.translate(src2, SVector(0, 1, 0))

trg = CompScienceMeshes.translate(src, SVector(2.0, 0, 0))
Γsrc = CompScienceMeshes.weld(src, src2)
Γtrg = trg

##