import numpy as np
import matplotlib.pyplot as plt

def TEModeField(a, b, m, n, f, r):
    c0 = 3e8
    mu = 4 * np.pi * 1e-7
    omega = 2 * np.pi * f
    k0 = omega / c0
    kc = np.sqrt((m * np.pi / a)**2 + (n * np.pi / b)**2)
    
    if k0 > kc:
        beta = np.sqrt(k0**2 - kc**2)
    else:
        beta = -1j * np.sqrt(kc**2 - k0**2)
        
    x, y, z = r
        
    Ex = (1j * omega * mu * n * np.pi / (kc**2 * b)) * \
         np.cos(m * np.pi * x / a) * np.sin(n * np.pi * y / b) * np.exp(-1j * beta * z)
    
    Ey = (-1j * omega * mu * m * np.pi / (kc**2 * a)) * \
         np.sin(m * np.pi * x / a) * np.cos(n * np.pi * y / b) * np.exp(-1j * beta * z)
    
    Ez = np.zeros_like(Ex)
    
    Hx = (1j * beta * m * np.pi / (kc**2 * a)) * \
         np.sin(m * np.pi * x / a) * np.cos(n * np.pi * y / b) * np.exp(-1j * beta * z)
    
    Hy = (1j * beta * n * np.pi / (kc**2 * b)) * \
         np.cos(m * np.pi * x / a) * np.sin(n * np.pi * y / b) * np.exp(-1j * beta * z)
    
    Hz = np.cos(m * np.pi * x / a) * np.cos(n * np.pi * y / b) * np.exp(-1j * beta * z)

    E = np.array([Ex, Ey, Ez])
    H = np.array([Hx, Hy, Hz])
    
    P_vec = np.cross(E, H, axis=0)
    P_magnitude = np.linalg.norm(P_vec, axis=0)
    
    return E, H, P_magnitude

if __name__ == "__main__":
    a = 2.286e-2
    b = 1.016e-2
    frequencies = [2e9, 5e9, 6e9, 7e9, 10e9, 15e9, 20e9]
    modes_to_test = [(1,0), (2,0), (0,1), (1,1)]
    
    
    x_cross = np.linspace(0, a, 101)
    y_cross = np.linspace(0, b, 101)
    X_c, Y_c = np.meshgrid(x_cross, y_cross)
    E_c, _, _ = TEModeField(a, b, 1, 0, 7e9, [X_c, Y_c, 0])
    E_magnitude = np.sqrt(np.sum(np.abs(E_c)**2, axis=0))
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_c*100, Y_c*100, E_magnitude, 
                        cmap='magma', 
                        edgecolor='none', 
                        antialiased=True)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='|E| Magnitude (V/m)')
    ax.set_title('3D Visualization of Electric Field Magnitude |E|\n(TE10 Mode at 7 GHz, z=0)')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_zlabel('|E| (V/m)')

    plt.show()
    
    x_vec = np.linspace(0, a, 100)
    z_vec = np.linspace(0, 5e-2, 100)
    X, Z = np.meshgrid(x_vec, z_vec)

    for frec in frequencies:
        if frec == 15e9:
            for (m_idx, n_idx) in modes_to_test:
                _, _, P_mag = TEModeField(a, b, m_idx, n_idx, frec, [X, b/2, Z])
                
                fig = plt.figure(figsize=(10, 7))
                ax = fig.add_subplot(111, projection='3d')
                surf = ax.plot_surface(Z*100, X*100, P_mag, cmap='viridis', edgecolor='none')
                
                ax.set_title(f'3D Power Flow: TE{m_idx}{n_idx} Mode at {frec/1e9} GHz')
                ax.set_xlabel('z (cm)')
                ax.set_ylabel('x (cm)')
                ax.set_zlabel('Power Density')
                plt.show()
        else:
            m_idx, n_idx = 1, 0
            _, _, P_mag = TEModeField(a, b, m_idx, n_idx, frec, [X, b/2, Z])
            
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(Z*100, X*100, P_mag, cmap='viridis', edgecolor='none')
            
            ax.set_title(f'3D Power Flow: TE{m_idx}{n_idx} Mode at {frec/1e9} GHz')
            ax.set_xlabel('z (cm)')
            ax.set_ylabel('x (cm)')
            ax.set_zlabel('Power Density')
            plt.show()