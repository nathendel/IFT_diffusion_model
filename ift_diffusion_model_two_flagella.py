from __future__ import division, print_function
import numpy as np
# import matplotlib.pyplot as plt
import scipy.stats as st
# import random
import time


'''
The difference between this simulation and the single-flagellum simulation is that this one simulated two flagella. It does this by keeping track of each flagellum separately.
In each time step, each flagellum goes through the simulation process including a potential injection event. They share a common pool of material, the variable "tubulin".
When a motor is injected, the amount of tubulin it takes with it is proportional (through the constant k_tub) to the remaining tubulin in the pool (tubulin-L0-L1).
A potential advancement of this program would be to generalize it to any number of flagella. I was more interested in just the two-flagellum case, so I hard coded in two.

The mechanics of this program are an extention of the simulation in ift_diffusion_model_nlh.py. That file has a more complete description in the comments on how it works.
This essentially does the same task but with two flagella that share building material from a common pool.l

Also, this uses an old naming scheme for variables, sorry for the confusion. The motor's property isactive means if it is in the flagellum, and isbound means if it is bound to the axoneme.
So if a motor is active and bound, it is in IFT. If it is active and not bound, it is in diffusion. If it is not active, it is in the base. I cleaned this up to be clearer in the one-flagellum 
file but not here.
'''

class Cell:
    cells = []

    # def __init__(self, t=0, L=0, N=200, trans_speed=1.8 / 10, k_on=.2, k_off=.2,
    #              avalanche_on=True, thresh=5, num_release=5,
    #              L_mod=True, build_size=.25 / 200, decay_size=.0037 / 10,
    #              rms_disp=3.5777 / 10, ava_power=2.85):  # L is length, N is number of particles

    def __init__(self, t=20000, L0=0, L1=0, N=400, trans_speed=2, k_on=0, k_off=0,
             avalanche_on=True, thresh=30, num_release=5,
             L_mod=True, build_size=.003, decay_size=.01, D=1.75, ava_power=2.85, ava_const=1, retro=False,
             L_hog=0,ss=False, t_step=.1, num_flagella=2, tubulin=30, k_tub = .000125):  # L is length, N is number of particles

        self.t = t
        self.L0 = L0
        self.L1 = L1
        # self.trans_speed = trans_speed
        self.k_on = k_on
        self.k_off = k_off
        self.N = N
        self.motors = [Motor(self, flagellum=round(i/(N-1))) for i in range(N)] #this line distributes the motors evenly between the two flagella
        self.ava_power=ava_power
        self.ava_const = ava_const
        # self.wholecell = wholecell

        self.active50 = self.N
        self.avalanche_on = avalanche_on
        self.thresh = thresh
        self.num_release = num_release
        self.recruited = 0
        self.L_mod = L_mod
        self.build_size = build_size
        # self.decay_size = decay_size
        # self.rms_disp = rms_disp
        self.ava_power = ava_power
        self.ava = []
        # self.avaT = np.zeros(t, num_flagella)
        # self.avaT=[]
        self.D = D # 1.75 from Alex Chien and Ahmet Yildiz
        self.L_hog = L_hog #hand of god -- change length manually

        self.L0_trace = np.zeros(t)
        self.L1_trace = np.zeros(t)
        # self.flux = np.zeros(t)
        # self.base = np.zeros(t)
        # self.N_diffuse = np.zeros(t)
        # self.track_active = np.zeros(t)

        self.retro=retro
        self.current_time=0
        self.ss=ss

        self.t_step = t_step #s

        self.trans_speed = t_step*trans_speed #2um, from Alex Chien and Ahmet Yildiz

        # self.D = D*1.75
        self.rms_disp = (2*D*self.t_step)**.5 #um
        # self.trans_speed = trans_speed*t_step #actually a distance
        self.decay_sizeMS = decay_size #m/s
        self.decay =decay_size #m/s
        self.decay_size = t_step * decay_size #meters in one time step
        self.build_size = build_size
        self.tubulin = tubulin
        self.k_tub = k_tub
        self.tubulin_in_IFT = 0 #this takes into account how much tubulin is in IFT and hasn't reached the tip yet. It solves the problem that when the new motor enters IFT, before it was only looking at the flagellar lengths when it calculates how much cargo to add

        self.L_predict = (2*self.D*(self.N-self.thresh)*self.build_size/self.decay_sizeMS)**.5

        if t:
            self.sim(self.t)

            if self.ss:
                while not self.is_steadystate():
                    # print('not ss')
                    self.extend(int(500/self.t_step))
            # self.time2ss = np.argmax(self.L_trace>(self.L-.1))
            # self.L=np.mean(self.L_trace[-3000:])
            # self.time2ss = np.argmax(self.L_trace>self.L)*self.t_step
            #calculate distribution of diffusing motors
            # density=[]
            # for i in range(self.current_time-10000,self.current_time):
                # dis=a.distr(i)
                # density+=self.distr(i)
            # self.kd = st.gaussian_kde(density)
            # self.diff_distr=self.kd.evaluate(np.linspace(0,self.L,100))



    def cut(self, flagellum=1):
        if flagellum==0:
            self.L0 = 0
        if flagellum==1:
            self.L1 = 0
        self.extend(20000)


    def count_active(self):
        return sum([p.isactive for p in self.motors])

    def extend(self,extend_time):
        # self.avaT = np.concatenate((self.avaT,np.zeros(extend_time)))
        self.L0_trace = np.concatenate((self.L0_trace,np.zeros(extend_time-1)))
        self.L1_trace = np.concatenate((self.L1_trace,np.zeros(extend_time-1)))
        # self.flux = np.concatenate((self.flux,np.zeros(extend_time-1,num_flagella)))
        # self.base = np.concatenate((self.base,np.zeros(extend_time-1,num_flagella)))
        # self.N_diffuse = np.concatenate((self.N_diffuse,np.zeros(extend_time-1,num_flagella)))
        # self.avaT = np.concatenate((self.avaT,np.zeros(extend_time-1,num_flagella)))
        for p in self.motors:
            p.track = np.concatenate((p.track,np.zeros(extend_time-1)))
            p.activetrack = np.concatenate((p.activetrack,np.zeros(extend_time-1)))
            p.boundtrack = np.concatenate((p.boundtrack,np.zeros(extend_time-1)))
        # self.sim(self=self,t=extend_time,start=self.t)
        self.sim(self.current_time+extend_time,start=self.current_time)
    #
    # def is_steadystate(self,fit_range=1000, eps=5e-6):
    #     fit_range = int(fit_range/self.t_step)
    #     if len(self.L_trace) < fit_range:
    #         return False
    #     slope,intercept = np.polyfit(range(fit_range),self.L_trace[-1*fit_range:],1)

        # return abs(self.L_trace[-1] - self.L_trace[-1*fit_range]) < .001

        # return abs(slope)<eps

    def distr(self,time=None):
        if time == None:
            time = self.current_time
        return [p.track[time] for p in self.motors if (p.activetrack[time] and not p.boundtrack[time])]

    # def sim(self, t):
    def sim(self,t,start=0):
        # t_step = .1 #s
        # self.rms_disp = (2*self.D*t_step)**.5 #um
        # self.rms_disp /= 10 #update to account for 1/10 s simulation JK I was multiplying by t_step
        for i in range(start,t):
            self.current_time=i
            if i==np.floor(t/2) and self.L_hog:
                self.L *= self.L_hog


            #         print(self.count_active())
            if self.avalanche_on:
                self.avalanche()

            if self.L_mod:
                if self.L0 >= self.decay_size:
                    self.L0 -= self.decay_size

                if self.L1 >= self.decay_size:
                    self.L1 -= self.decay_size

                # new may 27
                elif self.L0 < self.decay_size:
                    self.L0 = 0

                elif self.L1 < self.decay_size:
                    self.L1 = 0

                self.L0_trace[i] = self.L0
                self.L1_trace[i] = self.L1

            for p in self.motors:

                if p.isactive:
                    if p.isbound:
                        p.active_trans()
                    else:
                        p.diffuse()

                # p.track[i] = p.pos
                # p.activetrack[i]=p.isactive
                # p.boundtrack[i]=p.isbound

            # self.flux[i] = sum([1 for j in self.motors if (j.pos < 1 and j.isbound and j.isactive)])
            # self.base[i]= sum([1 for j in self.motors if not j.isactive])
            # self.N_diffuse[i] = sum([j.isactive and not j.isbound for j in self.motors])
            # self.track_active[i] = self.count_active()

    def avalanche(self):
        # distr = floor(1/np.random.power(self.ava_power))

        # num_inactive = self.N - self.count_active()
        inactive0 = [p for p in self.motors if ((not p.isactive) and p.flagellum==0)]
        num_inactive0 = len(inactive0)

        inactive1 = [p for p in self.motors if ((not p.isactive) and p.flagellum==1)]
        num_inactive1 = len(inactive1)
        
        cargo_this_tstep = 0
        
        if num_inactive0 > self.thresh:
            #             release = min(floor(1/np.random.power(3)),num_inactive)
            # distr = int(5 * np.random.weibull(1) + 1)  # parameters are totally arbitrary
            distr0 = int((num_inactive0-self.thresh+10) * np.random.weibull(self.ava_power) + self.ava_const)
            release0 = min(distr0, num_inactive0)
            # release = num_inactive #big avalanches
            # self.ava.append(release)
            # self.avaT.append(release)
            # self.avaT[self.current_time]=release

            for i in range(release0):  # commented out to try power law
                #             for i in range(1/np.random.power(3))
                inactive0[i].isactive = True
                inactive0[i].isbound = True
                inactive0[i].build_size = self.k_tub * (self.tubulin - self.L0 - self.L1 - self.tubulin_in_IFT)
                cargo_this_tstep += inactive0[i].build_size

                if self.L_mod:
                    inactive0[i].built = False
            # self.recruited += self.num_release

        # else:
            # self.avaT[self.current_time]=0
            # self.avaT.append(0)


        if num_inactive1 > self.thresh:
            #             release = min(floor(1/np.random.power(3)),num_inactive)
            # distr = int(5 * np.random.weibull(1) + 1)  # parameters are totally arbitrary
            distr1 = int((num_inactive1-self.thresh+10) * np.random.weibull(2.85) + 1)
            release1 = min(distr1, num_inactive1)
            # release = num_inactive #big avalanches
            # self.ava.append(release)
            # self.avaT.append(release)
            # self.avaT[self.current_time]=release

            for i in range(release1):  # commented out to try power law
                #             for i in range(1/np.random.power(3))
                inactive1[i].isactive = True
                inactive1[i].isbound = True
                inactive1[i].build_size = self.k_tub * (self.tubulin - self.L0 - self.L1 - self.tubulin_in_IFT)
                cargo_this_tstep += inactive1[i].build_size

                if self.L_mod:
                    inactive1[i].built = False
                    
        self.tubulin_in_IFT += cargo_this_tstep #this keeps track of how much tubulin is currently in IFT. This way the tubulin pool is changed when the motor is injected, not when the cargo is deposited at the tip.

    # def L_plot(self):
    #     plt.plot(self.L0_trace)
    #     plt.plot(self.L1_trace)
    #     plt.xlabel('time')
    #     plt.ylabel('length')
    #     plt.show()

    def __repr__(self):
        string = 'Cell of lengths %s, %s' %(self.L0,self.L1)
        return string


class Motor:
    instances = []

    def __init__(self, cell, isactive=False, isbound=False, flagellum=0, build_size=.00125):
        self.pos = 0
        self.isactive = isactive
        self.isbound = isbound
        Motor.instances.append(self)
        self.cell = cell
        # self.track = np.zeros(self.cell.t)
        # self.activetrack = np.zeros(self.cell.t)
        # self.boundtrack = np.zeros(self.cell.t)
        self.built = False
        self.flagellum = flagellum
        self.build_size = build_size

    def diffuse(self):
        if self.flagellum == 0:
            if self.pos > self.cell.L0:  # for length decay
                self.pos = self.cell.L0

            if self.pos == self.cell.L0:
                if not self.isbound:
                    self.pos -= self.cell.rms_disp

            else:
                r=np.random.rand()
                if r<.5:
                    self.pos -= self.cell.rms_disp
                else:
                    self.pos += self.cell.rms_disp

                if self.pos < 0:
                    self.pos = 0
                elif self.pos > self.cell.L0:
                    self.pos = self.cell.L0

            if self.pos <= 0:
                self.isactive = False  # keep this for later, using avalanche model
###
        elif self.flagellum == 1:
                if self.pos > self.cell.L1:  # for length decay
                    self.pos = self.cell.L1

                if self.pos == self.cell.L1:
                    if not self.isbound:
                        self.pos -= self.cell.rms_disp

                else:
                    r=np.random.rand()
                    if r<.5:
                        self.pos -= self.cell.rms_disp
                    else:
                        self.pos += self.cell.rms_disp

                    if self.pos < 0:
                        self.pos = 0
                    elif self.pos > self.cell.L1:
                        self.pos = self.cell.L1

                if self.pos <= 0:
                    self.isactive = False  # keep this for later, using avalanche model

    def active_trans(self):
        if self.flagellum == 0:
            if self.pos < self.cell.L0:
                self.pos += self.cell.trans_speed
                self.pos = min(self.pos, self.cell.L0)
            #         if self.pos == self.cell.L:
            if self.pos >= self.cell.L0:
                if not self.built:
                    self.cell.L0 += self.build_size
                    self.built = True
                    self.cell.tubulin_in_IFT -= self.build_size
                    # print(self.cell.tubulin_in_IFT)
                    
                self.isbound = False

        if self.flagellum == 1:
            if self.pos < self.cell.L1:
                self.pos += self.cell.trans_speed
                self.pos = min(self.pos, self.cell.L1)
            #         if self.pos == self.cell.L:
            if self.pos >= self.cell.L1:
                if not self.built:
                    self.cell.L1 += self.build_size
                    self.built = True
                    self.cell.tubulin_in_IFT -= self.build_size
                self.isbound = False




    def __repr__(self):
        string = 'Motor at position %s' % self.pos
        return string


#
# #
if __name__ == '__main__':
    a=Cell(t=10000)
#     print(a.L)
#     st=time.time()
#     b=Cell(t=5000,N=400)
#     while not b.is_steadystate():
#         # print('not ss')
#         b.extend(5000)
#     print(time.time()-st)
#
#     st2=time.time()
#     c=Cell(t=b.current_time, N=400)
#     print(time.time()-st2)

    # a.L_plot()
# 	# print(a.L)

## run profiler: python -m cProfile -s cumtime celL1.py


'''
notes on default params:
N=200 based on 10 transport complexes from Marshall and Rosenbaum 2001 Appendix multiplied by each injection event
sends 1-30 IFT particles from Ludington 2013 p.3926

decay_rate is from Marshall and Rosenbaum 2001 "intraflagellar transport balances continuous turnover.... appendix "Prediction of flagellar regeneration kinetics" section

'''
