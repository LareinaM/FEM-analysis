import numpy as np
import math
import igl
import meshplot as mp
import wildmeshing as wm
import polyfempy as pf
import scipy.sparse as sp
import random

name = "cuboid"
path = "data/"+name+"_l"+".obj"
out = "out/out_"+name+"_260"+".mesh"

test_save = 1

for i in range(10):
    """
    V, F = igl.read_triangle_mesh(path)
    #wm.tetrahedralize(path, out, mute_log=True, edge_length_r=1/10)
    """

    solver = pf.Solver()

    solver.load_mesh_from_path(out, vismesh_rel_area=1e-3)

    v, f = igl.read_triangle_mesh(out)

    minn = np.min(v, axis=0)
    maxx = np.max(v, axis=0)

    print(len(v))
    #nv = igl.per_vertex_normals(v, f)

    ps, ts, s = solver.get_boundary_sidesets()

    settings = pf.Settings()
    problem = pf.Problem()
    settings.set_pde(pf.PDEs.LinearElasticity)
    settings.set_material_params("E", 10000)
    settings.set_material_params("nu", 0.35)


    #fix = random.randint(1,6)
    fix = 2
    #           1: left       2: bottom    3: right      4: top        5: back       6: front
    face_idx = [[0,minn[0]], [1,minn[1]], [0, maxx[0]], [1, maxx[1]], [2, minn[2]], [2, maxx[2]]]

    force = random.randint(1,6)
    while force == fix:
        force = random.randint(1,6)
    mag = [random.randint(0,3),random.randint(0,3),random.randint(0,3)]

    # set the displacement value for the sideset id
    problem.set_displacement(fix, [0, 0, 0])
    problem.set_force(force, mag)
    settings.set_problem(problem)
    solver.settings(settings)
    solver.solve()

    p, tri, disp = solver.get_sampled_solution()

    m = np.linalg.norm(disp, axis=1)

    p_uni, indices, inverse = np.unique(p, return_index=True, return_inverse=True, axis=0)
    t_uni = np.array([inverse[tri[:, 0]], inverse[tri[:, 1]], inverse[tri[:, 2]], inverse[tri[:, 3]]]).transpose()
    d_uni = disp[indices, :]
    m_uni = m[indices]

    stress = solver.get_stresses()
    s_uni = stress[indices, :]

    mises, stress_ten = solver.get_sampled_mises_avg()
    mises_u = mises[indices, :]
    #print(mises)

    n,_ = s_uni.shape

    x = np.zeros([n,7])
    fix_ax = face_idx[fix-1][0]
    fix_val = face_idx[fix-1][1]

    force_ax = face_idx[force-1][0]
    force_val = face_idx[force-1][1]

    #print(fix,fix_ax,fix_val)
    print(force,mag)

    for i in range(n):
        pt = p_uni[i]
        if pt[fix_ax]==fix_val:
            x[i][0] = 1
        x[i][1] = pt[0]
        x[i][2] = pt[1]
        x[i][3] = pt[2]
        if pt[force_ax]==force_val and not x[i][0]==1:
            x[i][4] = mag[0]
            x[i][5] = mag[1]
            x[i][6] = mag[2]

    #print("displacement: \n",d_uni[:10],"\n")
    y = np.hstack((mises_u,d_uni*1000))


    save_y = "train/y/"+name+" "+str(fix)+" "+str(force)+str(mag)+".txt"
    save_adj = "train/adj/"+name+" "+str(fix)+" "+str(force)+str(mag)+".txt"
    save_x = "train/x/"+name+" "+str(fix)+" "+str(force)+str(mag)+".txt"
    test_y = "test/y/"+name+" "+str(fix)+" "+str(force)+str(mag)+".txt"
    test_adj = "test/adj/"+name+" "+str(fix)+" "+str(force)+str(mag)+".txt"
    test_x = "test/x/"+name+" "+str(fix)+" "+str(force)+str(mag)+".txt"

    """
    generate y(target) file
    [stress, disp_x, disp_y, disp_z] """
    if test_save == 0:
        doc1 = open(save_y,'w')
    else:
        doc1 = open(test_y,'w')
    for item in y:
        for i in item:
            print(i, file=doc1, end = ' ')
        print("", file=doc1)
    doc1.close()

    E = np.transpose([t_uni[:,0], t_uni[:,1]])
    E = np.append(E, np.transpose([t_uni[:,0], t_uni[:,2]]))
    E = np.append(E, np.transpose([t_uni[:,0], t_uni[:,3]]))
    E = np.append(E, np.transpose([t_uni[:,1], t_uni[:,2]]))
    E = np.append(E, np.transpose([t_uni[:,1], t_uni[:,3]]))
    E = np.append(E, np.transpose([t_uni[:,2], t_uni[:,3]]))
    l = len(E)
    E.resize((int(l/2),2))
    print(E.shape)
    E = np.unique(E, axis=0)
    print(E.shape)


    """generate edge(-> adj) file"""
    if test_save == 0:
        doc2 = open(save_adj,'w')
    else:
        doc2 = open(test_adj,'w')
    for item in E:
        for i in item:
            print(i, file=doc2, end = ' ')
        print("", file=doc2)
    doc2.close()

    """
    generate x(input feature) file
    [fix, x, y, z, force_x, force_y, force_z] """
    if test_save == 0:
        doc3 = open(save_x,'w')
    else:
        doc3 = open(test_x,'w')
    for item in x:
        for i in item:
            print(i, file=doc3, end = ' ')
        print("", file=doc3)
    doc3.close()

    #print(E.shape, y.shape)
    #plot = mp.plot(p_uni+d_uni, t_uni, mises_u, return_plot=True)