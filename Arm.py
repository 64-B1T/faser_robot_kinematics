from faser_math import *
from faser_plotting.Draw.Draw import *
import numpy as np
import scipy as sci
import scipy.linalg as ling
import random
import json
from os.path import dirname, basename, isfile

class Arm:
    S = 0
    Sbody= 0
    q = 0
    originalQ = 0
    orignalMee= 0
    originalS = 0
    originalW = 0
    Mee= 0
    _Mlinks= 0
    _Mhome= 0
    _Glinks= 0
    _Dims= 0
    _theta = 0
    _jaxes = 0
    Masses = 0
    baseT = 0
    eepos = 0
    #Think this should work

    #Converted to python - Liam
    def __init__(self, baseT, baseS, Mee, q, W = np.array([0])):
        """"Create a serial arm
        :param baseT: Base transform of Arm. tmobject
        :param baseS: Screw list of arm. Nx6 matrix.
        :param Mee: Initial end effector of arm.
        :param q: q list for arm
        :param W: W list for arm
        """
        #Configure the arm given the base screws, the base transform.
        self.cameras = []
        self.initialize(baseT, baseS, Mee, q)
        self._Mlinks = 0
        self._Glinks = 0
        self._Mhome = None
        self._Dims = None
        self.failcount = 0
        for i in range(0,baseS.shape[1]):
            self.Sbody[:,i] = fmr.Adjoint(self.Mee.inv().gTM()) @ self.S[:,i]
        self.reversable = False
        if W.shape[0] != 1:
            self.reversable = True
            self.reversed = False
            self._jaxes = W
            self.originalW = W
        self.originalS = self.S
        self.jointMins = np.ones((baseS.shape[1])) * np.pi * -2
        self.jointMaxs = np.ones((baseS.shape[1])) * np.pi * 2
        self.FK(np.zeros((self.S.shape[1])))

    def initialize(self, baseT, baseS, Mee, q):
        """
        Helper for Serial Arm. Should be called internally
        :param baseT: Base transform of Arm. tmobject
        :param baseS: Screw list of arm. Nx6 matrix.
        :param Mee: Initial end effector of arm.
        :param q: q list for arm
        """
        self.S = baseS
        self.originalSbody = np.copy(baseS)
        self.baseT = baseT
        self.originalQ = q
        self.q = np.zeros((3, baseS.shape[1]))
        self.Sbody = np.zeros((6, baseS.shape[1]))
        if q.size > 1:
            for i in range(0,baseS.shape[1]):
                self.q[0:3,i] = fsr.TrVec(baseT,q[0:3,i])
                #Convert TrVec
        for i in range(0,baseS.shape[1]):
            self.S[:,i] = fmr.Adjoint(baseT.gTM()) @ baseS[:,i]
            if q.size <= 1:
                [w,th,q,h] = fmr.TwistToScrew(self.S[:,i])
                #Convert TwistToScrew
                self.q[0:3,i] = q; # For plotting purposes
        self.EET = Mee
        self.Mee = baseT @ Mee
        self.eepos = self.Mee.copy()
        self.originalMee = self.Mee.copy()

    """
       _  ___                            _   _
      | |/ (_)                          | | (_)
      | ' / _ _ __   ___ _ __ ___   __ _| |_ _  ___ ___
      |  < | | '_ \ / _ \ '_ ` _ \ / _` | __| |/ __/ __|
      | . \| | | | |  __/ | | | | | (_| | |_| | (__\__ \
      |_|\_\_|_| |_|\___|_| |_| |_|\__,_|\__|_|\___|___/
    """
    #Converted to python -Liam
    def FK(self, theta):
        """
        Calculates the end effector position of the serial arm given thetas
        :param theta: The array of theta values for each joint
        """
        self._theta = fsr.AngleMod(theta.reshape(len(theta)))
        eepos = tm(fmr.FKinSpace(self.Mee.gTM(),self.S,theta))
        self.eepos = eepos
        return eepos

    #Converted to python - Liam
    def FKLink(self,theta,i):
        """
        Calculates the position of a given joint provided a theta list
        :param theta: The array of theta values for each joint
        :param i: The index of the joint desired, from 0
        """
        # Returns the TM of link i
        # Lynch 4.1
        endpos =  tm(fmr.FKinSpace(self._Mhome[i].TM,self.S[0:6,0:i],theta[0:i]))
        return endpos

    #Converted to python - Liam
    def IK(self,T, th0 = np.zeros(1), check = 1, level = 6):
        """
        Calculates joint positions of a serial arm. All parameters are optional except the desired end effector position
        :param T: Desired end effector position to calculate for
        :param th0: Intial theta guess for desired end effector position. Set to 0s if not provided.
        :param check: Whether or not the program should retry if the position finding fails
        :param level: number of recursive calls allowed if check is enabled
        :return: List of thetas, success boolean
        """
        if th0.size == 1:
            th0 = fsr.AngleMod(self._theta.reshape(len(self._theta)))
        theta,success = fmr.IKinSpace(self.S,self.Mee.gTM(),T.gTM(),th0,0.00000001,0.00000001)
        theta = fsr.AngleMod(theta)
        self._theta = theta
        if success:
            self.eepos = T
        else:
            if check == 1:
                i = 0
                while i < level and success == 0:
                    thz = np.zeros((len(self._theta)))
                    for j in range(len(thz)):
                        thz[j] = random.uniform(-np.pi, np.pi)
                    theta,success = fmr.IKinSpace(self.S,self.Mee.gTM(),T.gTM(),th0,0.00000001,0.00000001)
                    i = i + 1
                if success:
                    self.eepos = T
        return theta, success


    def constrainedIK(self, T, th0, check = 1, level = 6):
        """
        Calculates joint positions of a serial arm, provided rotational constraints on the Joints
        All parameters are optional except the desired end effector position
        Joint constraints are set through the jointMaxs and jointMins properties, and should be arrays the same size as the number of DOFS
        :param T: Desired end effector position to calculate for
        :param th0: Intial theta guess for desired end effector position. Set to 0s if not provided.
        :param check: Whether or not the program should retry if the position finding fails
        :param level: number of recursive calls allowed if check is enabled
        :return: List of thetas, success boolean
        """
        if not isinstance(T, tm):
            print(T)
            print("Attempted pass ^")
            return self._theta
        Slist = self.S.copy()
        if check == 1:
            self.failcount = 0
        M = self.Mee.copy()
        eomg = .001
        ev = .0001
        thetalist = self._theta.copy()
        i = 0
        maxiterations = 30

        try:
            thetalist, success = fmr.IKinSpaceConstrained(Slist, M.gTM(), T.gTM(), thetalist, eomg, ev, self.jointMins, self.jointMaxs, maxiterations)
        except:
            thetalist, success = self.constrainedIKNoFMR(Slist, M, T, thetalist, eomg, ev, maxiterations)
        if success:
            self.eepos = T
        else:
            if check == 1:
                i = 0
                while i < level and success == 0:
                    thz = np.zeros((len(self._theta)))
                    for j in range(len(thz)):
                        thz[j] = random.uniform(self.jointMins[j], self.jointMaxs[j])
                    try:
                        thetalist, success = fmr.IKinSpaceConstrained(Slist, M.gTM(), T.gTM(), thz, eomg, ev, self.jointMins, self.jointMaxs, maxiterations)
                    except Exception as e:
                        thetalist, success = self.constrainedIK(T, thz, check = 0)
                        disp("FMR Failure: " + str(e))
                    i = i + 1
                if success:
                    self.eepos = T
        if not success:
            if check == 0:
                self.failcount += 1
            else:
                #print("Total Cycle Failure")
                self.FK(np.zeros(len(self._theta)))
        else:
            if self.failcount != 0:
                print("Success + " + str(self.failcount) + " failures")
            self.FK(thetalist)

        return thetalist, success

    def constrainedIKNoFMR(self, Slist, M, T, thetalist, eomg, ev, maxiterations):
        """
        USed as a backup function for the standard constrained IK
        """
        Tsb = fmr.FKinSpace(M.gTM(), Slist, thetalist)
        Vs = np.dot(fmr.Adjoint(Tsb), fmr.se3ToVec(fmr.MatrixLog6(np.dot(fmr.TransInv(Tsb), T.gTM()))))
        #print(fmr.MatrixLog6(np.dot(fmr.TransInv(Tsb), T)), "Test")
        err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg \
              or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
        if np.isnan(Vs).any():
            err = True
        i = 0
        while err and i < maxiterations:
            thetalist = thetalist \
                        + np.dot(np.linalg.pinv(fmr.JacobianSpace(Slist, thetalist)), Vs)
            for j in range(len(thetalist)):
                if thetalist[j] < self.jointMins[j]:
                    thetalist[j] = self.jointMins[j]
                if thetalist[j] > self.jointMaxs[j]:
                    thetalist[j] = self.jointMaxs[j];
            i = i + 1
            Tsb = fmr.FKinSpace(M.gTM(), Slist, thetalist)
            Vs = np.dot(fmr.Adjoint(Tsb), \
                        fmr.se3ToVec(fmr.MatrixLog6(np.dot(fmr.TransInv(Tsb), T.gTM()))))
            err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg \
                  or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
            if np.isnan(Vs).any():
                err = True
        success = not err
        return thetalist, success

    def IKForceOptimal(self, T, th0, forcev, randomSample = 1000, mode = "MAX"):
        """
        Early attempt at creating a force optimization package for a serial arm.
        Absolutely NOT the optimial way to do this. Only works for overactuated arms.
        :param T: Desired end effector position to calculate for
        :param th0: Intial theta guess for desired end effector position. Set to 0s if not provided.
        :param forcev: Force applied to the end effector of the arm. Wrench.
        :param randomSample: number of samples to test to look for the most optimal solution
        :param mode: Set Mode to reduce. Max: Max Force. Sum: Sum of forces. Mean: Mean Force
        :return: List of thetas
        """
        thetas = []
        for i in range(randomSample):
            thz = np.zeros((len(self._theta)))
            for j in range(len(th0)):
                thz[j] = random.uniform(-np.pi, np.pi)
            thetas.append(thz)
        fthetas = []
        t_wren = np.cross(T[0:3].reshape((3)), forcev)
        wrench = np.array([t_wren[0], t_wren[1], t_wren[2], forcev[0], forcev[1], forcev[2]]).reshape((6,1))
        for i in range(len(thetas)):
            th, suc =self.IK(T, thetas[i])
            if suc and sum(abs(fsr.Error(self.FK(th), T))) < .0001:
                fthetas.append(th)
        maxForce = []
        for i in range(len(fthetas)):
            if mode == "MAX":
                maxForce.append(max(abs(self.StaticForces(fthetas[i], wrench))))
            elif mode == "SUM":
                maxForce.append(sum(abs(self.StaticForces(fthetas[i], wrench))))
            elif mode == "MEAN":
                maxForce.append(sum(abs(self.StaticForces(fthetas[i], wrench))) / len(fthetas))
        index = maxForce.index(min(maxForce))
        self._theta = fthetas[index]
        return fthetas[index]

    def IKMotion(self,T,th0):
        """
        This calculates IK by numerically moving the end effector
        from the pose defined by th0 in the direction of the desired
        pose T.  If the pose cannot be reached, it gets as close as
        it can.  This can sometimes return a better result than IK.
        An approximate explanation an be found in Lynch 9.2
        :param T: Desired end effector position to calculate for
        :param th0: Intial theta guess for desired end effector position.
        :return: theta, success, t, thall
        """
        Tstart = self.FK(th0)
        Tdir = T @ Tstart.inv()
        twdir = fsr.TwistFromTransform(Tdir)

        #[t,thall] = ode15s(@(t,x)(pinv(self.Jacobian(x))*twdir),[0 1],th0);
        res = lambda t, x: linalg.pinv(self.Jacobian(x))*twdir
        t, thall = sci.integrate.ode(res).set_integrator('vode', method='bdf', order=15)

        theta = thall[-1,:].conj().transpose()
        if fsr.Norm(T-self.FK(theta)) < 0.001:
            success = 1
        else:
            success = 0
        return (theta, success, t, thall)

    def IKFree(self,T,th0,inds):
        """
        Only allow th0(freeinds) to be varied
        Method not covered in Lynch.
        SetElements inserts the variable vector x into the positions
        indicated by freeinds in th0.  The remaining elements are
        unchanged.
        :param T: Desired end effector position to calculate for
        :param th0: Intial theta guess for desired end effector position.
        :param inds: Free indexes to move
        :return: theta, success, t, thall
        """
        #thetafree = fsolve(@(x)(obj.FK(SetElements(th0,freeinds,x))-T),th0(freeinds))
        res = lambda x : fsr.TAAtoTM(self.FK(fsr.SetElements(th0,inds,x))-T)
        #Use newton_krylov instead of fsolve
        thetafree = sci.optimize.fsolve(res,th0[inds])
        # thetafree = fsolve(@(x)(self.FK(SetElements(th0,freeinds,x))-T),th0(freeinds));
        theta = np.squeeze(th0);
        theta[inds] = np.squeeze(thetafree);
        if fsr.Norm(T-self.FK(theta)) < 0.001:
            success = 1;
        else:
            success = 0;
        return (theta,success)


    """
        _  ___                            _   _            _    _      _
       | |/ (_)                          | | (_)          | |  | |    | |
       | ' / _ _ __   ___ _ __ ___   __ _| |_ _  ___ ___  | |__| | ___| |_ __   ___ _ __ ___
       |  < | | '_ \ / _ \ '_ ` _ \ / _` | __| |/ __/ __| |  __  |/ _ \ | '_ \ / _ \ '__/ __|
       | . \| | | | |  __/ | | | | | (_| | |_| | (__\__ \ | |  | |  __/ | |_) |  __/ |  \__ \
       |_|\_\_|_| |_|\___|_| |_| |_|\__,_|\__|_|\___|___/ |_|  |_|\___|_| .__/ \___|_|  |___/
                                                                        | |
                                                                        |_|
    """

    def RandomPos(self):
        """
        Create a random position, return the end effector TF
        """
        thz = np.zeros((len(self._theta)))
        for j in range(len(thz)):
            thz[j] = random.uniform(self.jointMins[j], self.jointMaxs[j])
        pos = self.FK(thz)
        return pos


    def Reverse(self):
        """
        Flip around the serial arm so that the end effector is not the base and vice versa. Keep the same end pose
        """
        if not self.reversable:
            return
        old_thetas = np.copy(self._theta)
        new_theta = np.zeros((len(self._theta)))
        for i in range(self.S.shape[1]):
            new_theta[i] = old_thetas[len(old_thetas) - 1 - i]
        new_S = np.copy(self.originalS)
        new_EE = self.Mee.copy()
        new_BT = self.FK(self._theta)
        new_W = np.copy(self._jaxes)
        new_Q = np.copy(self.originalQ)
        for i in range(new_W.shape[1]):
            new_W[0:3,i] = self._jaxes[0:3, new_W.shape[1] - 1 - i]
        diffs = np.zeros((3, new_Q.shape[1]-1))
        for i in range(new_Q.shape[1]-1):
            diffs[0:3,i] = self.originalQ[0:3,self.originalQ.shape[1] - 1 - i] - self.originalQ[0:3,self.originalQ.shape[1] - 2 - i]
        #print(diffs, "diffs")
        for i in range(new_Q.shape[1]):
            if i == 0:
                new_Q[0:3,i] = self.originalQ[0:3, self.originalQ.shape[1] - 1] - np.sum(diffs,axis = 1)
            else:
                new_Q[0:3, i] = new_Q[0:3, i -1] + diffs[0:3, i - 1]
        for i in range(new_S.shape[1]):
            new_S[0:6,i] = np.hstack((new_W[0:3,i], np.cross(new_Q[0:3,i],new_W[0:3,i])))
        new_BT = new_BT @ tm(np.array([0, 0, 0, 0, np.pi, 0])) @ tm(np.array([0, 0, 0, 0, 0, np.pi]))
        if np.size(self._Dims) != 1:
            newDims = np.zeros((self._Dims.shape))
            for i in range(self._Dims.shape[1]):
                newDims[0:3,i] = self._Dims[0:3,self._Dims.shape[1] - i -1]
            self._Dims = newDims
        if len(self._Mhome) != 1:
            newMhome = [None] * len(self._Mhome)
            for i in range(len(newMhome)):
                newMhome[i] = self._Mhome[len(newMhome) - i -1]
            self._Mhome = newMhome
        self.S = new_S
        self.originalS = np.copy(new_S)
        #print(self.baseT, "")
        new_EE = new_BT @ self.EET
        self.baseT = new_BT
        self.originalQ = new_Q
        self.q = np.zeros((3, new_S.shape[1]))
        self.Sbody = np.zeros((6, new_S.shape[1]))
        if new_Q.size > 1:
            for i in range(0,new_S.shape[1]):
                self.q[0:3,i] = fsr.TrVec(new_BT, new_Q[0:3,i])
                #Convert TrVec
        for i in range(0,new_S.shape[1]):
            self.S[:,i] = fmr.Adjoint(new_BT.gTM()) @ new_S[:,i]
            if new_Q.size <= 1:
                [w,th,q,h] = fmr.TwistToScrew(self.S[:,i])
                #Convert TwistToScrew
                self.q[0:3,i] = q; # For plotting purposes
        self.Mee = new_EE
        self.originalMee = self.Mee.copy()
        if len(self._Mhome) != 1:
            Mi = [None] * len(self._Mhome)
            Mi[0] = self._Mhome[0];
            for i in range(1,6):
                Mi[i] = self._Mhome[i-1].inv() @ self._Mhome[i]
            Mi[len(self._Mhome) -1] = self._Mhome[5].inv() @ self.Mee
            self._Mlinks = Mi
        self._Glinks = 0
        for i in range(0,new_S.shape[1]):
            self.Sbody[:,i] = fmr.Adjoint(self.Mee.inv().gTM()) @ self.S[:,i]
        #print(new_theta)
        self.FK(new_theta)


    """
      __  __       _   _               _____  _                   _
     |  \/  |     | | (_)             |  __ \| |                 (_)
     | \  / | ___ | |_ _  ___  _ __   | |__) | | __ _ _ __  _ __  _ _ __   __ _
     | |\/| |/ _ \| __| |/ _ \| '_ \  |  ___/| |/ _` | '_ \| '_ \| | '_ \ / _` |
     | |  | | (_) | |_| | (_) | | | | | |    | | (_| | | | | | | | | | | | (_| |
     |_|  |_|\___/ \__|_|\___/|_| |_| |_|    |_|\__,_|_| |_|_| |_|_|_| |_|\__, |
                                                                           __/ |
                                                                          |___/
    """

    def lineTrajectory(self, target, initial = 0, execute = True, tol = np.array([.05, .05, .05, .05, .05, .05]), delt = .01):
        """
        Move the arm end effector in a straight line towards the target
        :param target: Target pose to reach
        :param intial: Starting pose. If set to 0, as is default, uses current position
        :param execute: Execute the desired motion after calculation
        :param tol: tolerances on motion
        :param delt: delta in meters to be calculated for each step
        """
        if initial == 0:
            initial = self.eepos.copy()
        satis = False
        init_theta = np.copy(self._theta)
        theta_list = []
        count = 0
        while not satis and count < 2500:
            count+=1
            error = fsr.Error(target, initial).gTAA().flatten()
            satis = True
            for i in range(6):
                if abs(error[i]) > tol[i]:
                    satis = False
            initial = fsr.CloseGap(initial, target, delt)
            theta_list.append(np.copy(self._theta))
            self.IK(initial,self._theta)
        self.IK(target, self._theta)
        theta_list.append(self._theta)
        if (execute == False):
            self.FK(init_theta)
        return theta_list


    def vServoSP(self, target, tol = 2, ax = 0, plt = 0, fig = 0):
        """
        Use a virtual camera to perform visual servoing to target
        :param target: Object to move to
        :param tol: piexel tolerance
        :param ax: matplotlib object to draw to
        :param plt: matplotlib plot
        :param fig: whether or not to draw
        :return: Thetalist for arm, figure object
        """
        if (len(self.cameras) == 0):
            print("NO CAMERA CONNECTED")
            return
        inTarg = False
        Done = False
        st = self.FK(self._theta)
        theta = 0
        j = 0
        plt.ion()
        images = []
        while not (inTarg and Done):
            for i in range(len(self.cameras)):
                tpos = tm()
                inTarg = True
                Done = True
                img, q, suc = self.cameras[i][0].getPhoto(target)
                if not suc:
                    print("Failed to locate Target")
                    return self._theta
                if img[0] < self.cameras[i][2][0] - tol:
                    tpos[0] = -.01
                    inTarg = False
                if img[0] > self.cameras[i][2][0] + tol:
                    tpos[0] = .01
                    inTarg = False
                if img[1] < self.cameras[i][2][1] - tol:
                    tpos[1] = -.01
                    inTarg = False
                if img[1] > self.cameras[i][2][1] + tol:
                    tpos[1] = .01
                    inTarg = False
                if inTarg:
                    d = fsr.Distance(self.eepos, target)
                    print(d)
                    if d < .985:
                        Done = False
                        tpos[2] = -.01
                    if d > 1.015:
                        Done = False
                        tpos[2] = .01
                st = st @ tpos
                theta = self.IK(st, self._theta)
                self.updateCams()
                if fig != 0:
                    ax = plt.axes(projection = '3d')
                    ax.set_xlim3d(-7,7)
                    ax.set_ylim3d(-7,7)
                    ax.set_zlim3d(0,8)
                    DrawArm(self, ax)
                    DrawRectangle(target, [.2, .2, .2], ax)
                    print("Animating")
                    plt.show()
                    plt.savefig('VideoTemp' + "/file%03d.png" % j)
                    ax.clear()
            j = j + 1
        return theta, fig

    def FollowTwistFixVels(self,th0,twist,fixinds,fixvels,tstart,tend):
        """
        Not implemented because I'm too dumb for a good ode45 conversion
        """
        #Fix ODE45
        #[t,th] = ode45(@(t,x)(obj.ThetadotSpaceFixVels(x,twist,fixinds,fixvels)),[tstart tend],th0);
        return None

    def GoToGoalEE(self, th0, Tend):
        """
        Legacy from ME5984
        """
        twnorm = fsr.TwistNorm(self.TwistEETransToGoalEE(th0,Tend))
        #Fix ODE45 Conversion
        #t, th = ode45(@(t,x)(obj.ThetadotEETrans(x,twnorm*obj.UnitTwistEETransToGoalEE(x,Tend))),[0 1],th0)

    def PDControlToGoalEE(self, theta, Tee, Kp, Kd, prevtheta, maxthetadot):
        """
        Legacy from ME5984
        """
        curEE = self.FK(theta)
        prevEE = self.FK(prevtheta)
        errEE = fsr.Norm(curEE[0:3,3]-Tee[0:3,3])
        derrEE_dt = errEE-fsr.Norm(prevEE[0:3,3]-Tee[0:3,3])
        scale = Kp @ errEE + Kd @ min(0, derrEE_dt)

        tw = self.TwistSpaceToGoalEE(theta,Tee)
        twnorm = fsr.TwistNorm(tw)
        utw = tw/twnorm
        thetadot = self.ThetadotSpace(theta,utw)
        scaledthetadot = maxthetadot/max(math.abs(thetadot)) @ thetadot @ scale
        return scaledthetadot



    """
       _____      _   _                                     _    _____      _   _
      / ____|    | | | |                    /\             | |  / ____|    | | | |
     | |  __  ___| |_| |_ ___ _ __ ___     /  \   _ __   __| | | (___   ___| |_| |_ ___ _ __ ___
     | | |_ |/ _ \ __| __/ _ \ '__/ __|   / /\ \ | '_ \ / _` |  \___ \ / _ \ __| __/ _ \ '__/ __|
     | |__| |  __/ |_| ||  __/ |  \__ \  / ____ \| | | | (_| |  ____) |  __/ |_| ||  __/ |  \__ \
      \_____|\___|\__|\__\___|_|  |___/ /_/    \_\_| |_|\__,_| |_____/ \___|\__|\__\___|_|  |___/

    """
    def saveArmState(fname):
        """
        unfinished
        """
        arm_dict = {"Slist" : self.S.copy(),
            "baseT" : self.baseT.copy(),
            "originalQ" : self.originalQ.copy(),
            "Qlist" : self.q.copy(),
            "SBody" : self.Sbody.copy(),
            "EETrans" : self.EET.copy(),
            "EEGlobal" : self.Mee.copy(),
            "EECurrent" : self.eepos.copy(),
            "MLinks" : self._Mlinks.copy(),
            "GLinks" : self._Glinks.copy()}
        with open(fname, 'w') as json_file:
            json.dump(arm_dict, json_file)

    def loadArmFromFile(fname):
        """
        unfinished
        """
        with open(fname) as f:
            data = json.load(f)
        self.baseT = data['baseT']
        self.originalQ = data['originalQ']
        self.q = data['Qlist']
        self.Sbody = data['sBody']
        self.EET = data['EETrams']
        self.Mee = data['EEGlobal']
        self.eepos = data['EECurrent']
        self._Mlinks = data["MLinks"]
        self._Glinks = data["GLinks"]

    def SetDynamicsProperties(self, _Mlinks = None, _Mhome = None, _Glinks = None, _Dims = None):
        """
        Set dynamics properties of the arm
        At mimimum Dims are a required parameter for drawing of the arm.
        :param _Mlinks: The mass matrices of links
        :param _MHome: List of Home Positions
        :param _GLinks: Mass Matrices (Inertia)
        :param _Dims: Dimensions of links
        """
        self._Mlinks = _Mlinks
        self._Mhome = _Mhome
        self._Glinks = _Glinks
        self._Dims = _Dims

    def SetMasses(self, mass):
        """
        set Masses
        :param mass: mass
        """
        self.Masses = mass

    def TestArmValues(self):
        """
        prints a bunch of arm values
        """
        np.set_printoptions(precision=4)
        np.set_printoptions(suppress=True)
        print("S")
        print(self.S, title = "Slist")
        print("Sbody")
        print(self.Sbody, title = "Sbody")
        print("Q")
        print(self.q, title = "q list")
        print("Mee")
        print(self.Mee, title = "Mee")
        print("originalMee")
        print(self.orignalMee, title = "originalMee")
        print("_Mlinks")
        print(self._Mlinks, title = "Link Masses")
        print("_Mhome")
        print(self._Mhome, title = "_Mhome")
        print("_Glinks")
        print(self._Glinks, title = "_Glinks")
        print("_Dims")
        print(self._Dims, title = "Dimensions")


    def getJointTransforms(self):
        """
        returns tmobjects for each link in a serial arm
        :return: tmlist
        """
        Dims = np.copy(self._Dims).conj().T
        retTAA = [None] * Dims.shape[0]

        Mee = self.baseT
        eepos = tm(fmr.FKinSpace(Mee.gTM(), self.S[0:6,0:0],self._theta[0:0]))
        #print(eepos, "EEPOS")
        retTAA[0] = eepos
        for i in range((self.S.shape[1])):
            if self._Mhome == None:
                t = tm(np.zeros((6)))
                t.TAA[0:3,0] = self.originalQ[0:3,i]
                t.TAAtoTM()
                Mee = self.baseT @ t
            else:
                Mee = self._Mhome[i]
            #print(Mee, "Mee" + str(i + 1))
            #print(self._theta[0:i+1])
            eepos = tm(fmr.FKinSpace(Mee.gTM(), self.S[0:6,0:i],self._theta[0:i]))
            #print(eepos, "EEPOS")
            retTAA[i] = eepos
        if Dims.shape[0] > self.S.shape[1]:
            #Fix handling of dims
            #print(fsr.TAAtoTM(np.array([0.0, 0.0, self._Dims[-1,2], 0.0 ,0.0, 0.0])))
            retTAA[len(retTAA) - 1] = self.FK(self._theta)
        return retTAA

    def SetArbitraryHome(self,theta,T):
        """
        #  Given a pose and some T in the space frame, find out where
        #  that T is in the EE frame, then find the home pose for
        #  that arbitrary pose
        """

        Tee = self.FK(theta)
        eeTarb = np.cross(np.inv(Tee),T)
        self.Mee = np.cross(self.Mee,eeTarb)

    #Converted to Python - Joshua
    def RestoreOriginalEE(self):
        """
        Retstore the original End effector of the Arm
        """
        self.Mee = self.originalMee



    """
      ______                                        _   _____                              _
     |  ____|                                      | | |  __ \                            (_)
     | |__ ___  _ __ ___ ___  ___    __ _ _ __   __| | | |  | |_   _ _ __   __ _ _ __ ___  _  ___ ___
     |  __/ _ \| '__/ __/ _ \/ __|  / _` | '_ \ / _` | | |  | | | | | '_ \ / _` | '_ ` _ \| |/ __/ __|
     | | | (_) | | | (_|  __/\__ \ | (_| | | | | (_| | | |__| | |_| | | | | (_| | | | | | | | (__\__ \
     |_|  \___/|_|  \___\___||___/  \__,_|_| |_|\__,_| |_____/ \__, |_| |_|\__,_|_| |_| |_|_|\___|___/
                                                                __/ |
                                                               |___/
    """

    def StaticForces(self,theta,wrenchEE):
        """
        Calculate forces on each link of the serial arm
        :param theta: Current position of the arm
        :param wrenchEE: end effector wrench (body frame)
        :return: forces in newtons on each joint
        """
        Tee = self.FK(theta) #Space Frame
        wrenchS = Tee.inv().Adjoint().conj().T @ wrenchEE
        tau = self.Jacobian(theta).conj().T @ wrenchS
        return tau

    def ForceMotion(self, thetas, wrench):
        """
        Unfinished
        """
        t_size = len(thetas)
        forceCurve = np.zeros((t_size, len(self._theta)))



    #def StaticForces(self, theta, wrenchEE):
    #    Tee = self.FK(theta)
    #    wrenchS = fmr.Adjoint(ling.inv(Tee)).conj().transpose() @ wrenchEE
    #    return self.Jacobian(theta).conj().transpose() @ wrenchS

    def StaticForcesInv(self,theta,tau):
        """
        Given a position on the arm and forces for each joint, calculate the wrench on the end effector
        :param theta: current joint positions of the arm
        :param tau: forces on the joints of the arm in Newtons
        :return: wrench on the end effector of the arm
        """
        x0 = np.zeros((len(theta)))
        temp = lambda x : (self.StaticForces(theta, x[0:6])-tau)
        wrenchEE = sci.optimize.fsolve(temp, x0)
        return wrenchEE[0:6]

    def InverseDynamics(self, theta, thetadot, thetadotdot, grav, wrenchEE):
        """
        Inverse dynamics
        """
        return self.InverseDynamicsE(theta, thetadot, thetadotdot, grav, wrenchEE)

    def InverseDynamicsEMR(self, theta, thetadot, thetadotdot, grav, wrenchEE):
        return fmr.InverseDynamics(theta, theta, thetadot, grav, wrenchEE, self._Mlinks, self._Glinks, self.S)

    def InverseDynamicsE(self, theta, thetadot, thetadotdot, grav, wrenchEE):
        #Multiple Bugs Fixed - Liam Aug 4 2019
        A = np.zeros((self.S.shape))
        V = np.zeros((self.S.shape))
        Vdot = np.zeros((self.S.shape))

        for i in range(self.S.shape[1]):
            #A[0:6,i] =(fmr.Adjoint(ling.inv(self._Mhome[i,:,:].reshape((4,4)))) @ self.S[0:6,i].reshape((6,1))).reshape((6))
            A[0:6,i] = (self._Mhome[i].inv().Adjoint() @ self.S[0:6,i]).reshape((6))

            #Ti_im1 = fmr.MatrixExp6(fmr.VecTose3(A[0:6,i]) * theta[i]) @ ling.inv(self._Mlinks[i,:,:])
            Ti_im1 = fmr.MatrixExp6(fmr.VecTose3(A[0:6,i]) * theta[i]) @ self._Mlinks[i].inv().TM
            if i > 0:
                V[0:6,i] = (A[0:6,i].reshape((6,1)) * thetadot[i] + fmr.Adjoint(Ti_im1) @ V[0:6,i-1].reshape((6,1))).reshape((6))
                #print(((A[0:6,i] * thetadotdot[i]).reshape((6,1)) + (fmr.Adjoint(Ti_im1) @ Vdot[0:6,i-1]).reshape((6,1)) + (fmr.ad(V[0:6,i]) @ A[0:6,i] * thetadot[i]).reshape((6,1))).reshape((6,1)), "vcomp")
                Vdot[0:6,i] = ((A[0:6,i] * thetadotdot[i]).reshape((6,1)) + (fmr.Adjoint(Ti_im1) @ Vdot[0:6,i-1]).reshape((6,1)) + (fmr.ad(V[0:6,i]) @ A[0:6,i] * thetadot[i]).reshape((6,1))).reshape((6))
            else:
                V[0:6,i] = (A[0:6,i].reshape((6,1)) * thetadot[i] + fmr.Adjoint(Ti_im1) @ np.zeros((6,1))).reshape((6))
                Vdot[0:6,i] = ((A[0:6,i] * thetadotdot[i]).reshape((6,1)) + (fmr.Adjoint(Ti_im1) @ np.vstack((np.array([[0],[0],[0]]), grav))).reshape((6,1)) + (fmr.ad(V[0:6,i]) @ A[0:6,i] * thetadot[i]).reshape((6,1))).reshape((6))
        F = np.zeros((self.S.shape))
        tau = np.zeros((theta.size,1))
        for i in range(self.S.shape[1]-1,-1,-1):
            if i == self.S.shape[1]-1:
                #continue
                Tip1_i = self._Mlinks[i+1].inv().TM
                F[0:6,i] = fmr.Adjoint(Tip1_i).conj().T @ wrenchEE + self._Glinks[i,:,:] @ Vdot[0:6,i] - fmr.ad(V[0:6,i]).conj().transpose() @ self._Glinks[i,:,:] @ V[0:6,i]
            else:
                #print( fmr.MatrixExp6(-fmr.VecTose3((A[0:6,i+1].reshape((6,1))) * theta(i + 1))) @ ling.inv(self._Mlinks[i+1,:,:]), "problem")
                Tip1_i = fmr.MatrixExp6(-fmr.VecTose3(A[0:6,i+1]) * theta[i + 1]) @ self._Mlinks[i+1].inv().TM
                F[0:6,i] = fmr.Adjoint(Tip1_i).conj().T @ F[0:6,i+1] + self._Glinks[i,:,:] @ Vdot[0:6,i] - fmr.ad(V[0:6,i]).conj().transpose() @ self._Glinks[i,:,:] @ V[0:6,i]

            tau[i] = F[0:6,i].conj().T @ A[0:6,i]
        return tau, A, V, Vdot, F

    def InverseDynamicsC(self, theta, thetadot, thetadotdot, grav, wrenchEE):
        n = theta.size
        A = np.zeros((6*n, n))
        G = np.zeros((6*n, n))
        for i in range (n):
            A[(i-1)*6+1:(i-1)*6+6,i] = fmr.Adjoint(ling.inv(self._Mhome[i,:,:])) @ self.S[0:6,i]
            G[(i-1)*6+1:(i-1)*6+6,(i-1)*6+1:(i-1)*6+7] = self._Glinks[i,:,:]
        W = np.zeros((6*n, 6*n))
        Vbase = np.zeros((6*n, 1))
        T10 = ling.inv(self.FKLink(theta,1))
        Vdotbase = np.hstack((self.Adjoint(T10) @ np.array([[0],[0],[0],[-grav]]),np.zeros((5*n,1))))
        Ttipend = ling.invv(self.FK(theta)) @ self.FKLink(theta, n)
        Ftip = np.vstack((np.zeros((5*n,1)), fmr.Adjoint(Ttipend).conj().transpose() @ wrenchEE))
        for i in range (1,n):
            Ti_im1 = ling.inv(self.FKlink(theta,i)) @ self.FKLink(theta,i-1)
            W[(i-1) * 6 + 1:(i-1) *6 + 6, (i-2)*6+1:(i-2)*6+6] = fmr.Adjoint(Ti_im1)
        L = ling.inv(np.identity((6*n))-W)
        V = L @ (A @ thetadot + Vbase)
        adV = np.zeros((6*n,6*n))
        adAthd = np.zeros((6*n,6*n))
        for i in range(1,n):
            adV[(i-1) * 6 + 1:(i-1) * 6+6,(i-1)*6+1:(i-1)*6+6] = fmr.ad(V[(i-1)*6+1:(i-1)*6+6,0])
            adAthd[(i-1)*6+1:(i-1) * 6 + 6, (i - 1) * 6 + 1 : (i - 1) * 6 + 6] = fmr. ad(thetadot[i] @ A[(i - 1) * 6 + 1 : (i - 1)* 6 + 6, i])
        Vdot = L @ (A @ thetadotdot - adAthd @ W @ V - adAthd @ Vbase @Vdotbase)
        F = L.conj().transpose() @ (G @ Vdot - adV.conj().transpose() @ G @ V + Ftip)
        tau = A.conj().transpose() @ F
        M = A.conj().transpose() @ L.conj().transpose() @ G @ L @ A

        return tau, M, c, G, ee

    def ForwardDynamicsE(self, theta, thetadot, tau, grav, wrenchEE):
        M = self.MasssMatrix(theta)
        h = self.CoriolisGravity(theta, thetadot, grav)
        ee = self.EndEffectorForces(theta,wrenchEE)
        thetadotdot = ling.inv(M) @ (tau-h-ee)

        return thetadotdot, M, h, ee

    def FordwardDynamics(self, theta, thetadot, tau, grav, wrenchEE):
        thetadotdot = fmr.ForwardDynamics(theta, thetadot, tau, grav, wrenchEE, self._Mlinks, self._Glinks, self.S)
        return thetadotdot

    def IntegrateForwardDynamics(self,theta0,thetadot0, tau,grav, wrenchEE,dt):
        #t,thetathetadot] = ode45(@(t,x)([x(7:12);obj.ForwardDynamicsE(x(1:6),x(7:12),tau,grav,wrenchEE)]),[0 dt],[theta0;thetadot0]);
        ## TODO:
        return None

    def MassMatrix(self, theta):
        #Debugged - Liam 8/4/19
        M = np.zeros(theta.size)
        for i in range(theta.size):
            Ji = self.JacobianLink(theta,i)
            jt = Ji.conj().T @ self._Glinks[i,:,:] @ Ji
            #M = M + jt
        #print(M, "M1")
        #print(fmr.MassMatrix(theta,self._Mlinks, self._Glinks, self.S), "Masses")
        return M

    def CoriolisGravity(self, theta, thetadot, grav):
        h = obj.InverseDynamicsE(theta,thetadot,0*theta,grav,np.zeros((6,1)))
        return h

    def EndEffectorForces(self,theta,wrenchEE):
        grav = np.array([[0.0],[0.0],[-9.81]])
        return self.InverseDynamicsE(theta,0*theta,0*theta,np.zeros((3,1)),wrenchEE)



    """
           _                 _     _                _____      _            _       _   _
          | |               | |   (_)              / ____|    | |          | |     | | (_)
          | | __ _  ___ ___ | |__  _  __ _ _ __   | |     __ _| | ___ _   _| | __ _| |_ _  ___  _ __  ___
      _   | |/ _` |/ __/ _ \| '_ \| |/ _` | '_ \  | |    / _` | |/ __| | | | |/ _` | __| |/ _ \| '_ \/ __|
     | |__| | (_| | (_| (_) | |_) | | (_| | | | | | |___| (_| | | (__| |_| | | (_| | |_| | (_) | | | \__ \
      \____/ \__,_|\___\___/|_.__/|_|\__,_|_| |_|  \_____\__,_|_|\___|\__,_|_|\__,_|\__|_|\___/|_| |_|___/

    """

    #Converted to Python - Joshua
    def Jacobian(self,theta):
        return fmr.JacobianSpace(self.S,theta)

    #Converted to Python - Joshua
    def JacobianBody(self,theta):
        return fmr.JacobianBody(self.Sbody,theta)

    #Converted to Python - Joshua
    #Fixed Bugs - Liam
    def JacobianLink(self,theta,i):
        t_ad = self.FKLink(theta,i).inv().Adjoint()
        t_js = fmr.JacobianSpace(self.S[0:6,0:i], theta[0:i])
        t_z = np.zeros((6,len(theta) - 1))
        t_mt = t_ad @ t_js
        return np.hstack((t_mt, t_z))

    def JacobianEE(self,theta):
        Js = self.Jacobian(theta)
        return (self.FK(theta).inv() @ Js).Adjoint()
        #return fmr.Adjoint()

    def JacobianEEtrans(self,theta):
        Tee = self.FK(theta)
        Tee[0:3,0:3] = np.identity((3))
        Js = self.Jacobian(theta)
        return fmr.Adjoint(ling.inv(Tee)) @ Js

    def NumericalJacobian(self, theta):
        Js = np.zeros((6,theta.size))
        temp = lambda x : np.reshape(self.FK(x),((1, 16)))
        Jst = fsr.NumJac(temp,theta,0.006)
        for i in range(0,np.size(theta)):
            Js[0:6,i] = fmr.se3ToVec(ling.inv(self.FK(theta).conj().transpose()) @ np.reshape(Jst[:,i],((4, 4))).conj().transpose())

        return Js


    """
       _____
      / ____|
     | |     __ _ _ __ ___   ___ _ __ __ _
     | |    / _` | '_ ` _ \ / _ \ '__/ _` |
     | |___| (_| | | | | | |  __/ | | (_| |
      \_____\__,_|_| |_| |_|\___|_|  \__,_|

    """


    def addCamera(self, cam, EEtoCam):
        cam.moveCamera(self.eepos @ EEtoCam)
        img, q, suc = cam.getPhoto(self.eepos @ tm(np.array([0, 0, 1, 0, 0, 0])))
        camL = [cam, EEtoCam, img]
        self.cameras.append(camL)
        print(self.cameras)

    def updateCams(self):
        for i in range(len(self.cameras)):
            self.cameras[i][0].moveCamera(self.eepos @ self.cameras[i][1])

    """
       _____ _                 __  __      _   _               _
      / ____| |               |  \/  |    | | | |             | |
     | |    | | __ _ ___ ___  | \  / | ___| |_| |__   ___   __| |___
     | |    | |/ _` / __/ __| | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
     | |____| | (_| \__ \__ \ | |  | |  __/ |_| | | | (_) | (_| \__ \
      \_____|_|\__,_|___/___/ |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    """

    def move(self, T, stationary = False):
        curpos = self.eepos.copy()
        curth = self._theta.copy()
        self.initialize(T, self.originalSbody, self.EET, self.originalQ)
        if stationary == False:
            self.FK(self._theta)
        else:
            self.IK(curpos, curth)

    def Draw(self, ax):
        DrawArm(self, ax)
    #Converted to python -Liam
