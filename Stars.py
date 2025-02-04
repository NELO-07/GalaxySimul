from Galaxy import Galaxy
import numpy as np

class Stars(Galaxy):

    def __init__(self, galmass, ahalo, vhalo, rhalo, galpos, galvel, diskSize, galtheta, galphi, n):
        super().__init__(galmass, ahalo, vhalo, rhalo, galpos, galvel)
        self.diskSize = diskSize
        self.galtheta = galtheta
        self.galphi   = galphi
        self.n = n

        self.starpos = np.full((3, self.n), 0.)
        self.starvel = np.full((3, self.n), 0.)
        self.staracc = np.full((3, self.n), 0.)

    def moveStars(self, dtime):
        newstarpos = self.starpos + self.starvel * dtime + 0.5 * self.staracc * (dtime**2)
        newstarvel = self.starvel + self.staracc * dtime

        self.starpos = newstarpos
        self.starvel = newstarvel

    def initStars(self, seed=None):
        if seed is not None:
            np.random.seed(seed)  # Set the random seed for reproducibility

        cosphi   = np.cos(self.galphi)
        sinphi   = np.sin(self.galphi)
        costheta = np.cos(self.galtheta)
        sintheta = np.sin(self.galtheta)
        for i in range(self.n):
            bad = True
            while bad:
                xtry = self.diskSize*(1.0-2.0*np.random.random())
                ytry = self.diskSize*(1.0-2.0*np.random.random())
                rtry = np.sqrt(xtry**2 + ytry**2)
                if rtry < self.diskSize:
                    bad = False

            ztry = 0.0
            xrot = xtry*cosphi + ytry*sinphi*costheta + ztry*sinphi*sintheta
            yrot = -xtry*sinphi + ytry*cosphi*costheta + ztry*cosphi*sintheta
            zrot = -ytry*sintheta + ztry*costheta
            rot  = np.array([xrot, yrot, zrot])
            self.starpos[:, i] = rot + self.galpos.reshape(-1)

            vcirc = np.sqrt(self.interiorMass(rtry)/rtry)

            vxtry = -vcirc*yrot/rtry
            vytry = vcirc*xrot/rtry
            vztry = 0.0

            vxrot = vxtry*cosphi + vytry*sinphi*costheta + vztry*sinphi*sintheta
            vyrot = -vxtry*sinphi + vytry*cosphi*costheta + vztry*cosphi*sintheta
            vzrot = -vytry*sintheta + vztry*costheta

            vrot = np.array([vxrot, vyrot, vzrot])
            self.starvel[:, i] = vrot + self.galvel.reshape(-1)
            self.staracc = np.full((1, 3), 0.0)

    def scaleMass(self, massFact):
        self.diskSize = self.diskSize * np.sqrt(massFact)
        super().scaleMass(massFact)
