#############################
# main_mpi.py (모든 스텝 저장)
#############################

import argparse
import numpy as np
import time
import os

from mpi4py import MPI
from Galaxy import Galaxy
from Stars import Stars
from Orbit import Orbit


class SimMPI:
    def __init__(self, args):
        self.args = args

    def makeGalaxy(self):
        galmass = 4.8
        ahalo   = 0.1
        vhalo   = 1.0
        rhalo   = 5.0
        diskSize= 2.5

        galpos = np.zeros((3,1))
        galvel = np.zeros((3,1))

        gal1theta = self.args.theta1 * (np.pi/180.)
        gal1phi   = self.args.phi1   * (np.pi/180.)
        gal2theta = self.args.theta2 * (np.pi/180.)
        gal2phi   = self.args.phi2   * (np.pi/180.)
        tot_nstar = self.args.tot_nstar
        mratio    = self.args.mratio

        n_gal1 = int(tot_nstar / (1.0 + mratio))
        n_gal2 = tot_nstar - n_gal1

        self.galaxy1 = Stars(galmass, ahalo, vhalo, rhalo,
                             galpos, galvel,
                             diskSize, gal1theta, gal1phi, n_gal1)
        self.galaxy2 = Stars(galmass, ahalo, vhalo, rhalo,
                             galpos, galvel,
                             diskSize, gal2theta, gal2phi, n_gal2)

        if self.args.big_halo:
            self.galaxy1.rhalo   = 20.0
            self.galaxy1.galmass = (self.galaxy1.vhalo**2 * self.galaxy1.rhalo**3) / \
                                   ((self.galaxy1.ahalo + self.galaxy1.rhalo)**2)

            self.galaxy2.rhalo   = self.galaxy2.rhalo * 4.0
            self.galaxy2.galmass = (self.galaxy2.vhalo**2 * self.galaxy2.rhalo**3) / \
                                   ((self.galaxy2.ahalo + self.galaxy2.rhalo)**2)

        self.galaxy2.scaleMass(mratio)

    def makeOrbit(self):
        energy = 0
        ecc    = 1
        rperi  = 3.0
        tperi  = self.args.peri

        self.crashOrbit = Orbit(
            energy, rperi, tperi, ecc,
            self.galaxy1.galmass, self.galaxy2.galmass,
            self.galaxy1.galpos, self.galaxy2.galpos,
            self.galaxy1.galvel, self.galaxy2.galvel
        )

        self.galaxy1.setPosvel(self.crashOrbit.p1pos, self.crashOrbit.p1vel)
        self.galaxy2.setPosvel(self.crashOrbit.p2pos, self.crashOrbit.p2vel)

    def runSimMPI(self):
        """
        모든 스텝에서:
          1) 은하 중심(rank=0) -> bcast
          2) 별 업데이트(각 rank)
          3) Gather & 저장 (매 스텝)
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        dt    = self.args.dt
        nstep = self.args.nstep

        # 1) rank=0에서만 별 초기화
        if rank == 0:
            if self.args.seed_fix:
                self.galaxy1.initStars(seed=111203)
                self.galaxy2.initStars(seed=111203)
            else:
                self.galaxy1.initStars()
                self.galaxy2.initStars()

            starpos1 = self.galaxy1.starpos
            starvel1 = self.galaxy1.starvel
            starpos2 = self.galaxy2.starpos
            starvel2 = self.galaxy2.starvel
            n_gal1 = self.galaxy1.n
            n_gal2 = self.galaxy2.n
        else:
            starpos1 = None
            starvel1 = None
            starpos2 = None
            starvel2 = None
            n_gal1 = 0
            n_gal2 = 0

        # bcast로 별 개수 정보 전파
        n_gal1 = comm.bcast(n_gal1, root=0)
        n_gal2 = comm.bcast(n_gal2, root=0)

        # 균등 분할
        size1 = n_gal1 // size
        size2 = n_gal2 // size

        start1 = rank * size1
        end1   = (rank+1)*size1 if rank != size-1 else n_gal1
        start2 = rank * size2
        end2   = (rank+1)*size2 if rank != size-1 else n_gal2

        local_n1 = end1 - start1
        local_n2 = end2 - start2

        local_starpos1 = np.zeros((3, local_n1))
        local_starvel1 = np.zeros((3, local_n1))
        local_starpos2 = np.zeros((3, local_n2))
        local_starvel2 = np.zeros((3, local_n2))

        # Scatter
        comm.Barrier()
        if rank == 0:
            local_starpos1[:] = starpos1[:, start1:end1]
            local_starvel1[:] = starvel1[:, start1:end1]
            local_starpos2[:] = starpos2[:, start2:end2]
            local_starvel2[:] = starvel2[:, start2:end2]

            for r in range(1, size):
                i1 = r * size1
                i2 = (r+1)*size1 if r!=(size-1) else n_gal1
                j1 = r * size2
                j2 = (r+1)*size2 if r!=(size-1) else n_gal2

                comm.Send(starpos1[:, i1:i2].copy(), dest=r, tag=10)
                comm.Send(starvel1[:, i1:i2].copy(), dest=r, tag=11)
                comm.Send(starpos2[:, j1:j2].copy(), dest=r, tag=12)
                comm.Send(starvel2[:, j1:j2].copy(), dest=r, tag=13)
        else:
            comm.Recv(local_starpos1, source=0, tag=10)
            comm.Recv(local_starvel1, source=0, tag=11)
            comm.Recv(local_starpos2, source=0, tag=12)
            comm.Recv(local_starvel2, source=0, tag=13)

        comm.Barrier()

        # 2) 메인 루프 (매 스텝 저장)
        for step in range(nstep):
            # rank=0 은하 상호작용 -> bcast
            if rank == 0:
                dist = 3.5 * np.linalg.norm(self.galaxy1.galpos - self.galaxy2.galpos)
                self.galaxy1.galacc = self.galaxy2.acceleration(self.galaxy1.galpos)
                self.galaxy1.galacc += self.galaxy2.dynFriction(
                    self.galaxy1.interiorMass(dist/3.5),
                    self.galaxy1.galpos,
                    self.galaxy1.galvel
                )
                self.galaxy2.galacc = self.galaxy1.acceleration(self.galaxy2.galpos)
                self.galaxy2.galacc += self.galaxy1.dynFriction(
                    self.galaxy2.interiorMass(dist/3.5),
                    self.galaxy2.galpos,
                    self.galaxy2.galvel
                )
                comacc = ((self.galaxy1.galmass * self.galaxy1.galacc) +
                          (self.galaxy2.galmass * self.galaxy2.galacc)) / \
                         (self.galaxy1.galmass + self.galaxy2.galmass)
                self.galaxy1.galacc -= comacc
                self.galaxy2.galacc -= comacc

                self.galaxy1.moveGalaxy(dt)
                self.galaxy2.moveGalaxy(dt)

                data_bcast = {
                    'gal1pos': self.galaxy1.galpos,
                    'gal1vel': self.galaxy1.galvel,
                    'gal1acc': self.galaxy1.galacc,
                    'gal2pos': self.galaxy2.galpos,
                    'gal2vel': self.galaxy2.galvel,
                    'gal2acc': self.galaxy2.galacc
                }
            else:
                data_bcast = None

            data_bcast = comm.bcast(data_bcast, root=0)
            if rank != 0:
                self.galaxy1.galpos = data_bcast['gal1pos']
                self.galaxy1.galvel = data_bcast['gal1vel']
                self.galaxy1.galacc = data_bcast['gal1acc']
                self.galaxy2.galpos = data_bcast['gal2pos']
                self.galaxy2.galvel = data_bcast['gal2vel']
                self.galaxy2.galacc = data_bcast['gal2acc']

            # 별 가속도 & 업데이트
            acc_local_1 = (self.galaxy1.acceleration(local_starpos1) +
                           self.galaxy2.acceleration(local_starpos1))
            acc_local_2 = (self.galaxy1.acceleration(local_starpos2) +
                           self.galaxy2.acceleration(local_starpos2))

            local_starpos1 += local_starvel1 * dt + 0.5 * acc_local_1 * (dt**2)
            local_starvel1 += acc_local_1 * dt

            local_starpos2 += local_starvel2 * dt + 0.5 * acc_local_2 * (dt**2)
            local_starvel2 += acc_local_2 * dt

            # 매 스텝 Gather & 저장
            if rank == 0:
                gal1_starpos_global = np.zeros((3, n_gal1))
                gal2_starpos_global = np.zeros((3, n_gal2))

                gal1_starpos_global[:, start1:end1] = local_starpos1
                gal2_starpos_global[:, start2:end2] = local_starpos2

                for r in range(1, size):
                    i1 = r * size1
                    i2 = (r+1)*size1 if r!=(size-1) else n_gal1
                    j1 = r * size2
                    j2 = (r+1)*size2 if r!=(size-1) else n_gal2

                    buf1 = np.zeros((3, i2-i1))
                    buf2 = np.zeros((3, j2-j1))
                    comm.Recv(buf1, source=r, tag=100+r)
                    comm.Recv(buf2, source=r, tag=200+r)

                    gal1_starpos_global[:, i1:i2] = buf1
                    gal2_starpos_global[:, j1:j2] = buf2

                # 저장
                output_dir = "outputs"
                os.makedirs(output_dir, exist_ok=True)

                max_stars = max(n_gal1, n_gal2)
                pad1 = np.pad(gal1_starpos_global, ((0,0),(0, max_stars-n_gal1)), constant_values=np.nan)
                pad2 = np.pad(gal2_starpos_global, ((0,0),(0, max_stars-n_gal2)), constant_values=np.nan)
                frame = np.stack([pad1, pad2], axis=0)

                fname = os.path.join(output_dir, f"snapshot_t{step:04d}.npy")
                np.save(fname, frame)
            else:
                comm.Send(local_starpos1, dest=0, tag=100+rank)
                comm.Send(local_starpos2, dest=0, tag=200+rank)

        if rank == 0:
            print("All steps saved. Simulation done.")


def main():
    parser = argparse.ArgumentParser(description="Galaxy Collision Simulation (MPI, save every step)")
    parser.add_argument("--theta1",    type=float, default=0.)
    parser.add_argument("--phi1",      type=float, default=315.)
    parser.add_argument("--theta2",    type=float, default=0.)
    parser.add_argument("--phi2",      type=float, default=45.)
    parser.add_argument("--tot_nstar", type=int,   default=10000)
    parser.add_argument("--mratio",    type=float, default=3.0)
    parser.add_argument("--peri",      type=float, default=5.0)
    parser.add_argument("--dt",        type=float, default=0.04)
    parser.add_argument("--nstep",     type=int,   default=2000)
    parser.add_argument("--big_halo",  action="store_true")
    parser.add_argument("--seed_fix",  action="store_true")

    start_time = time.time()
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    sim = SimMPI(args)
    if rank == 0:
        print("[Rank 0] Setting up galaxies & orbit...")

    sim.makeGalaxy()
    sim.makeOrbit()

    if rank == 0:
        print("[Rank 0] Start simulation in parallel (EVERY STEP SAVE) ...")

    sim.runSimMPI()

    end_time = time.time()
    if rank == 0:
        print(f"Total time: {end_time - start_time:.3f} sec")


if __name__ == "__main__":
    main()
