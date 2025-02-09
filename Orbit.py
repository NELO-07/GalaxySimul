import numpy as np

class Orbit:

    def __init__(self, energy, rp, tp, eccentricity, m1, m2, p1pos, p2pos, p1vel, p2vel):
        self.energy = energy
        self.rp = rp
        self.tp = tp
        self.ecc= eccentricity
        self.m1 = m1
        self.m2 = m2
        self.p1pos = p1pos
        self.p2pos = p2pos
        self.p1vel = p1vel
        self.p2vel = p2vel
        self.initOrbit()

    def initOrbit(self):
        mu = self.m1 + self.m2

        p = 2*self.rp
        nhat = np.sqrt(mu/(p**3))
        cots = 3.0 * nhat * self.tp
        s = np.arctan(1.0/cots)
        cottheta = (1.0/(np.tan(s/2.0)))**(1/3)
        theta = np.arctan(1.0/cottheta)
        tanfon2 = 2.0/np.tan(2.0*theta)
        r = (p/2.0)*(1+tanfon2**2)

        vel = np.sqrt(2.0*mu/r)
        sinsqphi = p/(2.0*r)
        phi = np.arcsin(np.sqrt(sinsqphi))
        f = 2.0*np.arctan(tanfon2)
        xc = -r*np.cos(f)
        yc = r*np.sin(f)
        vxc = vel*np.cos(f+phi)
        vyc = -vel*np.sin(f+phi)
        xcom = self.m2 * xc / mu
        ycom = self.m2 * yc / mu
        vxcom = self.m2 * vxc / mu
        vycom = self.m2 * vyc / mu

        self.p1pos = np.array([[-xcom], [-ycom], [0.0]])
        self.p1vel = np.array([[-vxcom], [-vycom], [0.0]])
        self.p2pos = np.array([[xc-xcom], [yc-ycom], [0.0]])
        self.p2vel = np.array([[vxc-vxcom], [vyc-vycom], [0.0]])
