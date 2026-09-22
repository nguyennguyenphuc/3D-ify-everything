import numpy as np
from studio.geometry import project,unproject,train_intrinsics,camera_to_viewer


def test_unproject_project_round_trip():
    K=np.array([[100.,0,3],[0,110.,2],[0,0,1]])
    E=np.array([[1.,0,0,.2],[0,1,0,-.1],[0,0,1,.3]])
    depth=np.full((5,7),2.)
    xyz=unproject(depth,K,E)
    yy,xx=np.indices(depth.shape)
    pixels,z=project(xyz.reshape(-1,3),K,E)
    assert np.allclose(pixels,np.stack([xx,yy],-1).reshape(-1,2))
    assert np.allclose(z,2)


def test_resize_intrinsics_and_viewer_inverse():
    K=np.array([[300.,0,259.],[0,400.,259.],[0,0,1]])
    meta={'original_to_model':[[.5,0,-.25],[0,.5,-.25],[0,0,1]],'original_to_train':[[.25,0,-.375],[0,.25,-.375],[0,0,1]]}
    got=train_intrinsics(K,meta)
    # Pixel-center transforms retain the half-pixel convention through both scales.
    assert np.allclose(got,[[150,0,129.25],[0,200,129.25],[0,0,1]])
    E=np.array([[1.,0,0,1],[0,1,0,2],[0,0,1,3]])
    # Translation is the inverse world→camera translation; the local y/z axes flip.
    assert np.allclose(camera_to_viewer(E),np.array([[1,0,0,-1],[0,-1,0,-2],[0,0,-1,-3],[0,0,0,1]]))


def test_cross_view_rejects_inconsistent_depth():
    from studio.geometry import consistent_points
    K=np.eye(3)
    E=np.column_stack([np.eye(3),np.zeros(3)])
    points=np.array([[0.,0.,2.],[2.,0.,2.]])
    depths=np.array([[[2.,2.]],[[2.,8.]]])
    keep=consistent_points(points,0,depths,np.ones_like(depths),np.ones_like(depths,dtype=bool),[E,E],[K,K])
    assert keep.tolist()==[True,False]
