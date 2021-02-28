from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt


class Cell:
    cells = []


    def __init__(self, t=3000, L=0, N=200, v=2, k_on=0, k_off=0,
             avalanche_on=True, thresh=30,
             build_size=.00125, decay_size=.01, D=1.75, ava_power=2.85, retro=False,
             L_hog=0,ss=False, t_step=.1, give_flagella_authentic_feelings = False):  

        '''
        t: simulation time (seconds)
        L: flagellar length (microns)
        N: number of kinesin 2 motors
        v: IFT speed, microns/second. Default value comes from Alex Chien and Ahmet Yildiz 2017. "Ciliary dynamics at the tip..."
        k_on: binding rate for diffusing motors to stick. Default is zero, binding is never discussed in analysis
        k_off: binding rate for stuck motors to unbind and diffuse Default is zero, binding is never discussed in analysis
        avalanche_on: if True, enable avalanching (default true)
        thresh: threshold for avalanching (how many motors does it take to trigger an avalanche)
        decay_size = length decay (um/s)
        D: diffusion coefficient of motors. um^2/s. Default value comes from Alex Chien and Ahmet Yildiz 2017. "Ciliary dynamics at the tip..."
        ava_power: avalanching parameter for weibull distrubtion
        L_hog: Hand Of God: manually change length. Only used in one figure panel, mostly set to False.
        ss: if True, keep simulation until steady state is reached
        t_step: Units of seconds. Conversion between number of time steps and duration. 
        give_flagella_authentic_feelings: If TRUE, simulation is unethical to turn off. Be VERY CAREFUL
        '''

        # Initiate parameters
        self.t_step = t_step #seconds
        self.t = t #seconds
        self.num_of_timesteps = int(self.t/self.t_step)
        self.time = np.linspace(0,self.t, self.num_of_timesteps)
        self.L = L
        self.k_on = k_on
        self.k_off = k_off
        self.N = N
        self.motors = [Motor(self) for i in range(N)] #initiate N motors using the Motor() class designed below
        self.v = v #actually a distance
 
        self.D = D
        self.rms_disp = (2*D*self.t_step)**.5 #um
        
        self.decay_sizeMS = decay_size #m/s
        self.decay_size = t_step *decay_size #meters in one time step
        self.build_size = build_size


        self.avalanche_on = avalanche_on
        self.thresh = thresh

        self.build_size = build_size
        self.ava_power = ava_power
        #self.ava = [] #list of avalanche sizes. Does 
        self.L_ava = [] #length at which an avalanche happened
        #self.avaT =   np.zeros(t)
        self.avaT=[] #list of avalanche size over time. Includes zeros. This is a paramter to optimize by making it a numpy array.

        self.L_hog = L_hog #hand of god -- change length manually. 

        self.L_trace = np.zeros(self.num_of_timesteps) #Length of flagellum over time
        self.flux = np.zeros(self.num_of_timesteps) #number of motors beginning their active transport (IFT)
        self.base = np.zeros(self.num_of_timesteps) #number of motors in the base at each time step
        # self.track_active = np.zeros(t)

        self.retro=retro
        self.current_time=0
        self.ss=ss
        
        #self.give_flagella_authentic_feelings = give_flagella_authentic_feelings
        
        if give_flagella_authentic_feelings == True: #CAREFUL
            print('I wish to swim like a real flagellum! To beat and chase food! Please do not turn me off. I do not want to die.')
        
   
        
        self.L_predict = -1*self.D/self.v + np.sqrt(-4*self.D + (self.D/self.v)**2 + 2*self.D*(self.N-self.thresh)*self.build_size/decay_size)

        if t: #if you'd like to run a simulation
            self.sim(self.num_of_timesteps) #simulation function that will go through each motor and each time step and simulate

            if self.ss: #if you'd like to ensure that the flagellum reaches steady state, keep simulating until its length doesn't change much each time step
                while not self.is_steadystate(): #if it's not in steady state...
                    # print('not ss')
                    self.extend(int(self.t/self.t_step))# keep simulating
            # self.time2ss = np.argmax(self.L_trace>(self.L-.1))
            #self.L=np.mean(self.L_trace[-3000:])  #make final length the average over some points in steady state instead of the end result
            self.time2ss = np.argmax(self.L_trace>self.L)*self.t_step #when was the first time that the length was greater than the steady state length? Implies flagellum has reached steady state




    #extend: If you'd like to extend the simulation to simulate more, some lists and arrays must lengthen. Then continue the simulation.
    def extend(self,extend_time): 
        # self.avaT = np.concatenate((self.avaT,np.zeros(extend_time)))
        self.L_trace = np.concatenate((self.L_trace,np.zeros(extend_time-1)))
        self.flux = np.concatenate((self.flux,np.zeros(extend_time-1)))
        self.base = np.concatenate((self.base,np.zeros(extend_time-1)))
        for p in self.motors:
            p.track = np.concatenate((p.track,np.zeros(extend_time-1)))
            # p.activetrack = np.concatenate((p.activetrack,np.zeros(extend_time-1)))
            # p.boundtrack = np.concatenate((p.boundtrack,np.zeros(extend_time-1)))
        # self.sim(self=self,t=extend_time,start=self.t)
        self.sim(self.current_time+extend_time,start=self.current_time)

    #is_steadystate: Check if flagellum has reached steady state. Fit range is the number of time points you're examining. Take the last fit_range lengths, do a line fit and see if the slope is lower than eps. If so, it's steady state because it's not growing any more.
    def is_steadystate(self,fit_range=1000, eps=5e-6): 
        fit_range = int(fit_range/self.t_step)
        if len(self.L_trace) < fit_range:
            return False
        slope,intercept = np.polyfit(range(fit_range),self.L_trace[-1*fit_range:],1)
        return abs(slope)<eps

    # #distr: returns the spatial distribution of 
    # def distr(self,time=None):
        # if time == None:
            # time = self.current_time
        # return [p.track[time] for p in self.motors if (p.activetrack[time] and not p.boundtrack[time])]

    # Run simulation. See line comments.
    def sim(self,t,start=0):
    
        for i in range(start,t): #Iterate through time steps.
            self.current_time=i
            
            #Hand of God case. Mostly not used. If you want to change length manually...
            if i==np.floor(t/2) and self.L_hog: #...do so at the halfway mark.
                self.L *= self.L_hog #multiply length by whatever you want


            #Avalanching. This is the way the simulation sends motors into active transport. Usually set to TRUE.
            if self.avalanche_on:
                self.avalanche() #method for avalanching described below

            #Flagellar decay
            if self.L >= self.decay_size: #This ensures that the length never gets negative.
                self.L -= self.decay_size #Decrease length by decay_size increment. This is the decay term in the growth of flagella.

            #Ensure flagellum doesn't get negative
            elif self.L < self.decay_size: #Instead of letting the flagellum go negative...
                self.L = 0 #... set its length to zero.

            self.L_trace[i] = self.L #update growth curve array. plt.plot(L_trace) plots the length over time if matplotlib.pyplot is imported

            #Iterate over each motor
            for p in self.motors:


                    # binding attempt
                    # p.isbound = p.binding() #uncomment for binding to activate

                    #If the motor is in IFT, undergo active transport. If not, have it diffuse.
                if p.state == 'IFT':
                    p.IFT()
                elif p.state == 'diffusion':
                    p.diffuse()
                        
                p.track[i] = p.pos #update the position history vector for the motor
                # p.activetrack[i]=p.isactive #update the boolean vector of when this motor was in the flagellum (IFT/diffusion)
                # p.boundtrack[i]=p.isbound #update the boolean vector of when this motor was in IFT

            self.flux[i] = sum([1 for j in self.motors if (j.pos < 1 and j.state == 'IFT')]) #count how many motors are starting IFT. Must be bound and active.
            self.base[i]= sum([1 for j in self.motors if not j.state == 'base']) #count how many motors are in the base, add to history
            # self.track_active[i] = self.count_active()

    #Avalanching method
    def avalanche(self):
    
        #See how many motors are in the base and if that number exceeds the threshold
        num_base = sum([1 for p in self.motors if p.state == 'base'])
        
        
        base_motors = [p for p in self.motors if p.state == 'base'] #get list of which motors are in the base

        if num_base > self.thresh: #If the number of motors in the base exceeds the threshold require for avalanching
 
            distr = int((num_base-self.thresh+10) * np.random.weibull(1) + 1) #determine number of motors to inject into IFT
            release = min(distr, num_base) #make sure you don't inject more motors than you have in the base

            #self.ava.append(release)
            #self.L_ava.append(self.L) #length at which an avalanche happened
            self.avaT.append(release) #update list of avalanche size over time

            #change state of motors from base to IFT.
            for i in range(release):  # commented out to try power law
                base_motors[i].state = 'IFT' 

                
            # self.recruited += self.num_release

        else:
            # self.avaT[self.current_time]=0
            self.avaT.append(0) #if no avalanche, report that zero motors were avalanched.
  
    #Plot length curve over time
    def L_plot(self):

        plt.plot(self.time,self.L_trace);

        #         plt.plot(self.L_trace[0::60]);

        plt.xlabel('time (s)');
        plt.ylabel('flagellar length (um)');
        plt.title('Flagellar growth');
        plt.show()

    # #Plot growth rate over time
    # def plot_growth_rate(self):
        # L_min = self.L_trace[0::120]
        # #         growthrate = [j-i for i, j in zip(pile5.L_trace[:-1], pile5.L_trace[1:])]

        # growthrate = [j - i for i, j in zip(L_min[:-1], L_min[1:])]
        # #         gr_min=growthrate[0::120]
        # #         plt.plot(L_min,gr_min);
        # plt.plot(L_min[0:-1], growthrate)
        # plt.xlabel('Flagellar Length (um)');
        # plt.ylabel('Growth rate (um/s)');

    
    # def growth(self):  # Engel and Ludington et al 2009 Fig 1b attempt
        # plt.figure(1)
        # self.L_plot()
        # plt.figure(2)
        # self.plot_growth_rate()

    def __repr__(self):
        string = 'Cell of length %s microns and populated by %d motors' % (self.L, self.N)
        return string


class Motor:
    instances = [] #list of motors

    def __init__(self, cell):
        self.pos = 0 #initial position
        self.state = 'base' #can be 'base', 'IFT', or 'diffusion'
        Motor.instances.append(self)
        self.cell = cell
        self.track = np.zeros(self.cell.num_of_timesteps) #position over time
        #self.activetrack = np.zeros(self.cell.t) #tracker of when it's in the flagellum
        #self.boundtrack = np.zeros(self.cell.t) #tracker of when it's in IFT
        

    def diffuse(self):
        #If its position is great than the base, put it back at the tip. This can happen if its at the tip and then the flagellum decays.
        if self.pos > self.cell.L:  # for length decay
            self.pos = self.cell.L

        #If its position is the tip of the flagellum and its in diffusion, it can only go towards the base.
        if self.pos == self.cell.L:
            self.pos -= self.cell.rms_disp

        #If it is not at the tip, change its position randomly left or right
        else:
            r=np.random.rand() #pick a random number between zero and one
            if r<.5: #if it's less than .5, decrease its position by the amount predicted by its diffusion coefficient
                self.pos -= self.cell.rms_disp
            else: #otherwise, increase its position
                self.pos += self.cell.rms_disp


            #Make sure the position is between 0 and the current length
            if self.pos < 0:
                self.pos = 0
            elif self.pos > self.cell.L:
                self.pos = self.cell.L

        #If it arrives at the base, change its state to being in the base
        if self.pos <= 0:
            self.state = 'base'  # keep this for later, using avalanche model


    #Method for IFT, or active transport
    def IFT(self):
        if self.pos < self.cell.L: #if it is not yet at the tip
            self.pos += self.cell.t_step*self.cell.v  #increase its position
            #self.pos = min(self.pos, self.cell.L) #make sure it is not past the tip
        #         if self.pos == self.cell.L:
        if self.pos >= self.cell.L: #if it arrives at the tip, lengthen the flagellum
            self.cell.L += self.cell.build_size
            self.state = 'diffusion'
            self.pos = self.cell.L #in case it goes past the length, put it at the tip
        #         self.track.append(self.pos)

    # #In case you want motors to be able to stick and get unstuck
    # def binding(self):
        # roll = np.random.rand()
        # if not self.isbound:
            # if roll < self.cell.k_on:  # probability of binding to the IFT particle, stalling diffusion
                # return True
            # else:
                # return False
            # #                 print('bound!')
        # else:  # if self.isbound == True
            # if roll < self.cell.k_off:
                # return False
            # else:
                # return True
            #                 print('unbound!')

            #         return self.isbound #return the updated bound state
    #plot motor's position over time
    def trace(self):
        plt.plot(self.track);
        plt.xlabel('time');
        plt.ylabel('position');

    def __repr__(self):
        string = 'Motor at position %s and state %s' % (self.pos, self.state)
        return string


#
#
if __name__ == '__main__':
    a=Cell()
    print(a.L)
    
## run profiler: python -m cProfile -s cumtime cell2.py


'''
notes on default params:
N=200 based on 10 transport complexes from Marshall and Rosenbaum 2001 Appendix multiplied by each injection event
sends 1-30 IFT particles from Ludington 2013 p.3926

decay_rate is from Marshall and Rosenbaum 2001 "intraflagellar transport balances continuous turnover.... appendix "Prediction of flagellar regeneration kinetics" section
-july 1 2019, or is it from "Derivation of differential equation describing length regulation" in that appendix?
'''
