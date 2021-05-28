from faser_math import *
from faser_utils.disp.disp import *
import numpy as np
import scipy as sci
import scipy.linalg as ling
import copy
import json

class SP:

    def __init__(self, bJoints, tJoints, bT, tT, lmin, lmax, bThickness, tThickness, name):
        self.bottomJoints = np.copy(bJoints)
        self.topJoints = np.copy(tJoints)
        self.bPos = self.bottomJoints.conj().transpose()
        self.pPos = self.topJoints.conj().transpose()
        self._bottomT = bT.copy()
        self._topT = tT.copy()
        self._bJS = np.zeros((3,6))
        self._tJS = np.zeros((3,6))

        #Debug
        self.legSafety = .001
        self.debug = 0

        #Physical Parameters
        self.bThickness = bThickness
        self.tThickness = tThickness
        if lmin == 0:
            self.lmin = 0
            self.lmax = 2
        self.lmin = lmin
        self.lmax = lmax

        #Reserve Val
        self.poffset = fsr.Distance(bT,tT)
        self.poffmat = tm([0, 0, self.poffset, 0, 0, 0])

        #Drawing Characteristics
        self.OuterTopRad = 0
        self.OuterBotRad = 0
        self.ShaftRad = 0
        self.MotorRad = 0

        #Empty array indicates these values haven't been populated yet
        self.legForces =  np.zeros(1)
        self.topWEE =  np.zeros(1)
        self.bottomWEE =  np.zeros(1)

        #Mass values from bottom mass, top mass, and actuator portion masses can be set directly.
        self.bPlateMass = 0
        self.tPlateMass = 0
        self.actuatorTopMass = 0
        self.actuatorBottomMass = 0
        self.actuatorTN =0
        self.actuatorBN =0
        self.plateTN =0
        self.plateBN =0
        self.grav = 9.81
        self.dir = np.array([0, 0, -1])
        self.ActTopCOG = 0
        self.ActBotCOG = 0
        self.force_limit= 0

        #Tolerances and Limits
        self.jointDeflectionMax = 140/2*np.pi/180#2*np.pi/5
        self.plateRotationLimit = np.cos(60*np.pi/180)

        #Newton Settings
        self.tol_f = 1e-5/2
        self.tol_a = 1e-5/2
        self.maxIters = 1e4

        #Errors and Counts
        self.failcount = 0
        self.validation_settings = [1, 0, 0, 1]
        self.FKmode = 1
        self.validation_error = ""

        self.IK(bT, tT, protect = True)

        self.baseAngles = self._tJS.T.copy()
        self.bottomAngles = self._bJS.T.copy()

        self.baseAngles = [None] * 6
        self.bottomAngles = [None] * 6
        for i in range(6):
            self.baseAngles[i] = fsr.GlobalToLocal(self.gbottomT(), tm([self._tJS.T[i][0],self._tJS.T[i][1],self._tJS.T[i][2],0,0,0]))
            self.bottomAngles[i] = fsr.GlobalToLocal(self.gtopT(), tm([self._bJS.T[i][0],self._bJS.T[i][1],self._bJS.T[i][2],0,0,0]))

        t1 = fsr.GlobalToLocal(self.gtopT() @ tm([0, 0, -self.tThickness, 0, 0, 0]), tm([self._tJS[0,0], self._tJS[1,0], self._tJS[2,0], 0,0,0]))
        t2 = fsr.GlobalToLocal(self.gtopT() @ tm([0, 0, -self.tThickness, 0, 0, 0]), tm([self._tJS[0,2], self._tJS[1,2], self._tJS[2,2], 0,0,0]))
        t3 = fsr.GlobalToLocal(self.gtopT() @ tm([0, 0, -self.tThickness, 0, 0, 0]), tm([self._tJS[0,4], self._tJS[1,4], self._tJS[2,4], 0,0,0]))
        self.reorients = [t1, t2, t3]


        #Compatibility
        self.plateThickness = (self.tThickness + self.bThickness) / 2
        self.plateTransform = tm([0, 0, self.plateThickness, 0, 0, 0])

        #Validation Settings


    """
       _____      _   _                                     _    _____      _   _
      / ____|    | | | |                    /\             | |  / ____|    | | | |
     | |  __  ___| |_| |_ ___ _ __ ___     /  \   _ __   __| | | (___   ___| |_| |_ ___ _ __ ___
     | | |_ |/ _ \ __| __/ _ \ '__/ __|   / /\ \ | '_ \ / _` |  \___ \ / _ \ __| __/ _ \ '__/ __|
     | |__| |  __/ |_| ||  __/ |  \__ \  / ____ \| | | | (_| |  ____) |  __/ |_| ||  __/ |  \__ \
      \_____|\___|\__|\__\___|_|  |___/ /_/    \_\_| |_|\__,_| |_____/ \___|\__|\__\___|_|  |___/

    """
    def SetMasses(self, plateMass, actuatorTop, actuatorBottom, grav = 9.81, tPlateMass = 0):
        """
        Set masses for each SP in the Assembler, note that because central platforms share plates, these weights are halved with respect to end plates
        Takes in plateMass, actuator shaft mass, actuator bottom mass, and acceleration due to gravity
        """
        self.bPlateMass = plateMass
        if tPlateMass != 0:
            self.tPlateMass = tPlateMass
        else:
            self.tPlateMass = plateMass
        self.SetGrav(grav)
        self.actuatorTopMass = actuatorTop
        self.actuatorBottomMass = actuatorBottom
        self.actuatorBN = self.actuatorBottomMass * self.grav
        self.actuatorTN = self.actuatorTopMass * self.grav
        self.plateTN = self.tPlateMass * self.grav
        self.plateBN = self.bPlateMass * self.grav

    def SetGrav(self, grav = 9.81):
        self.grav = grav

    def SetCOG(self, motor_grav_center, shaft_grav_center):
        self.ActTopCOG = shaft_grav_center
        self.ActBotCOG = motor_grav_center

    def SetAngleDev(self, MaxAngleDev = 55):
        self.jointDeflectionMax = MaxAngleDev*np.pi/180

    def SetPlateAngleDev(self, MaxPlateDev = 60):
        self.plateRotationLimit = np.cos(MaxPlateDev * np.pi / 180)

    def SetDrawingDimensions(self, OuterTopRad, OuterBotRad, ShaftRad, MotorRad):
        self.OuterTopRad = OuterTopRad
        self.OuterBotRad = OuterBotRad
        self.ShaftRad = ShaftRad
        self.MotorRad = MotorRad

    def _setPlatePos(self, bottomT, topT):
        if bottomT is not None:
            self._bottomT = bottomT
        if topT is not None:
            self._topT = topT

    def gLens(self):
        """
        returns leg lengths
        """
        return self.lengths.copy()

    def gtopT(self):
        """
        Return the transform of the top plate
        """
        return self._topT.copy()

    def gbottomT(self):
        """
        Return the transform of the bottom plate
        """
        return self._bottomT.copy()

    def GetActuatorUnit(self, p1, p2, dist):
        v1 = np.array([p1[0], p1[1], p1[2]])
        unitb = (np.array([p2[0], p2[1], p2[2]]) - v1)
        unit = unitb / ling.norm(unitb)
        pos = v1 + (unit * dist)
        return tm([pos[0], pos[1], pos[2], 0, 0, 0])

    def getActuatorLoc(self, num, type = 'm'):
        """
        Returns the position of a specified actuator. Takes in an actuator number and a type.
        m for actuator midpoint
        b for actuator motor position
        t for actuator top position
        """

        pos = 0
        if type == 'm':
            pos = np.array([(self._bJS[0,num] + self._tJS[0,num])/2,
                (self._bJS[1,num] + self._tJS[1,num])/2,
                (self._bJS[2,num] + self._tJS[2,num])/2])
        bleg = tm([self._bJS[0,num], self._bJS[1,num],self._bJS[2,num],0,0,0])
        tleg = tm([self._tJS[0,num], self._tJS[1,num],self._tJS[2,num],0,0,0])
        if type == 'b':
            #return fsr.TMMidRotAdjust(bleg, bleg, tleg, mode = 1) @ tm([0, 0, self.ActBotCOG, 0, 0, 0])
            return self.GetActuatorUnit(bleg, tleg, self.ActBotCOG)
        if type == 't':
            #return fsr.TMMidRotAdjust(tleg, tleg, bleg, mode = 1) @ tm([0, 0, self.ActTopCOG, 0, 0, 0])
            return self.GetActuatorUnit(tleg, bleg, self.ActTopCOG)
        newPos = tm([pos[0], pos[1], pos[2], 0, 0, 0])
        return newPos

    def SpinCustom(self, rot):
        oldBase = self.gbottomT()
        sp.move(tm())
        topCop = self.gtopT()
        ModTop = self._tJS.copy()
        ModBot = self._bJS.copy()
        ModTopHs = self.topJoints[2,0:6]
        ModBotHs = self.bottomJoints[2,0:6]
        rotTM = tm([0, 0, 0, 0, 0, rot * np.pi / 180])
        self.move(rotTM)
        ModTopNew = self._tJS.copy()
        ModBotNew = self._bJS.copy()
        ModTop[0:2,0:6] = ModTopNew[0:2,0:6]
        ModBot[0:2,0:6] = ModBotNew[0:2,0:6]
        ModBot[2,0:6] = ModBotHs
        ModTop[2,0:6] = ModTopHs
        self.move(tm())
        self.bottomJoints = ModBot
        self.topJoints = ModTop
        self._bJS = ModBotNew
        self._tJS = ModTopNew
        self.move(oldBase)

    def IKPath(self, goal, steps):
        return fsr.IKPath(self.gtopT(), goal, steps)

    def IK(self, bottomT = None, topT = None, protect = False, dir = 1):

        bottomT, topT = self._bottomTopCheck(bottomT, topT)

        lens, bottomT, topT = self.IKHelper(bottomT, topT, protect, dir)
        #Determine current transform


        self._bottomT = bottomT.copy()
        self._topT = topT.copy()

        #Ensure a valid position
        valid = True
        if not protect:
            valid = self.validate()
        return lens, valid

    def IKHelper(self, bottomT = None, topT = None, protect = False, dir = 1):
        """
        Calculates Inverse Kinematics for a single stewart plaform.
        Takes in bottom plate transform, top plate transform, protection paramter, and direction
        """
        #If not supplied paramters, draw from stored values
        bottomT, topT = self._bottomTopCheck(bottomT, topT)

        #Check for excessive rotation
        #Poses which would be valid by leg length
        #But would result in singularity


        #Set bottom and top transforms
        #self._bottomT = bottomT
        #self._topT = topT




        #Call the IK method from the JIT numba file (FASER HIGH PER)
        #Shoulda just called it HiPer FASER. Darn.
        self.lengths, self._bJS, self._tJS = fmr.SPIKinSpace(bottomT.gTM(), topT.gTM(), self.bottomJoints, self.topJoints, self._bJS, self._tJS)
        self.curTrans = fsr.GlobalToLocal(bottomT, topT)
        return np.copy(self.lengths), bottomT, topT

    def FK(self, L, bottomT =None, reverse = False, protect = False):
        """
        FK Host Function
        Takes in length list,
        optionally bottom position, reverse parameter, and protection
        """
        #FK host function, calls subfunctions depedning on the value of FKmode
        #return self.FKSciRaphson(L, bottomT, reverse, protect)
        #bottomT, n = self._applyPlateTransform(bottomT = bottomT)
        if self.FKmode == 0:
            bottom, top = self.FKSolve(L, bottomT, reverse, protect)
        else:
            bottom, top = self.FKRaphson(L, bottomT, reverse, protect)

        if not self._continuousTranslationConstraint():
            if self.debug:
                disp("FK Resulted In Inverted Plate Alignment. Repairing...")
            #self.IK(topT = self.gbottomT() @ tm([0, 0, self.poffset, 0, 0, 0]))
            #self.FK(L, protect = True)
            self.fixUpsideDown()
        self.curTrans = fsr.GlobalToLocal(bottom, top)
        #self._undoPlateTransform(bottom, top)
        valid = True
        if not protect:
            valid = self.validate()
        return top, valid

    def FKSciRaphson(self, L, bottomT = None, reverse = False, protect = False):
        """
        Use Python's Scipy module to calculate forward kinematics. Takes in length list,
        optionally bottom position, reverse parameter, and protection
        """
        L = L.reshape((6, 1))
        mag = lambda x : abs(x[0]) + abs(x[1])+ abs(x[2]) + abs(x[3]) + abs(x[4]) + abs(x[5])
        fk = lambda x : mag(self.IKHelper(bottomT, tm(x), protect = True)[0] - L).flatten()
        jac = lambda x : (self.InverseJacobianSpace(bottomT, tm(x)))
        x0 = (self.gbottomT() @ self.poffmat).TAA.flatten()

        root = sci.optimize.minimize(fk, x0).x
        #disp(root, "ROOT")
        self.IK(bottomT, tm(root), protect = True)
        return bottomT, tm(root)

    def SimplifiedRaphson(self, L, bottomT = None, reverse = False, protect = False):
        """
        Follow the method in the Parallel Robotics Textbook
        """
        tol_f = 1e-4;
        tol_a = 1e-4;
        #iteration limits
        maxIters = 1e4

        if bottomT == None:
            bottomT = self._bottomT

        x = self.gtopT().copy()
        iter = 0
        success = False
        while not success and iter < maxIters:
            x = x + self.InverseJacobianSpace(bottomT, x ) @ (L - self.IK(topT = x, protect = protect))
            x.AngleMod()
            #disp(x)
            if np.all(abs(x[0:3]) < tol_f) and np.all(abs(x[3:6]) < tol_a):
                success = True
            iter+=1

        if iter == maxIters:
            print("Failed to Converge")

        return tm(x)



    def FKSolve(self, L, bottomT = None, reverse = False, protect = False):
        """
        Older version of python solver, no jacobian used. Takes in length list,
        optionally bottom position, reverse parameter, and protection
        """
        #Do SPFK with scipy inbuilt solvers. Way less speedy or accurate than Raphson, but much simpler to look at
        L = L.reshape((6,1))
        self.lengths = L.reshape((6,1)).copy()
        #jac = lambda x : self.InverseJacobianSpace(topT = x)

        #Slightly different if the platform is supposed to be "reversed"
        if reverse:
            if bottomT == None:
                topT = self.gtopT()
            else:
                topT = bottomT
            fk = lambda x : (self.IK(tm(x), topT, protect = True) - L).reshape((6))
            sol = tm(sci.optimize.fsolve(fk, self.gtopT().gTAA()))
            #self._topT = bottomT
        else:
            #General calls will go here.
            if bottomT == None:
                #If no bottom pose is supplied, use the last known.
                bottomT = self.gbottomT()
            #Find top pose that produces the desired leg lengths.
            fk = lambda x : (self.IKHelper(bottomT, tm(x), protect = True)[0] - L).reshape((6))
            sol = tm(sci.optimize.fsolve(fk, self.gtopT().TAA))
            #self._bottomT = bottomT

        #If not "Protected" from recursion, call IK.
        if not protect:
            self.IK(protect = True)
        return bottomT, sol


    def FKRaphson(self, L, bottomT =None, reverse = False, protect = False):
        """
        Adapted from the work done by
        #http://jak-o-shadows.github.io/electronics/stewart-gough/stewart-gough.html
        Takes in length list,
        optionally bottom position, reverse parameter, and protection
        """
        if self.debug:
            disp("Starting Raphson FK")
        #^Look here for the original code and paper describing how this works.
        if bottomT == None:
            bottomT = self.gbottomT()
        success = True
        L = L.reshape((6))
        self.lengths = L.reshape((6,1)).copy()

        bottomTb = bottomT.copy()# @ tm([0, 0, self.bThickness, 0, 0, 0])
        bottomT = np.eye(4)
        #bottomT = bottomTb.copy()
        #newton-raphson tolerances
        #iteration limits
        iterNum = 0

        #Initial Guess Position
        #a = fsr.TMtoTAA(bottomT @ fsr.TM([0, 0, self.poffset, 0, 0, 0])).reshape((6))
        #disp(a, "Attempt")
        try:
            #ap = (fsr.LocalToGlobal(tm([0, 0, self.poffset, 0, 0, 0]), tm()))
            ap = (fsr.LocalToGlobal(self.curTrans, tm())).gTAA().reshape((6))
            a = np.zeros((6))
            for i in range(6):
                a[i] = ap[i]

            #Call the actual algorithm from the high performance faser library
            #Pass in initial lengths, guess, bottom and top plate positions, max iterations, tolerances, and minimum leg lengths
            a, iterNum = fmr.SPFKinSpaceR(bottomT, L, a, self.bPos, self.pPos, self.maxIters, self.tol_f, self.tol_a, self.lmin)

            #If the algorithm failed, try again, but this time set initial position to neutral
            if iterNum == self.maxIters:

                a = np.zeros((6))
                a[2] = self.poffset
                a, iterNum = fmr.SPFKinSpaceR(bottomT, L, a, self.bPos, self.pPos, self.maxIters, self.tol_f, self.tol_a, self.lmin)
                if iterNum == self.maxIters:
                    if self.debug:
                        print("Raphson Failed to Converge")
                    self.failcount += .1
                    self.IK(bottomTb, bottomTb @ self.poffmat, protect = True)
                    return self.gbottomT(), self.gtopT()

            #Otherwise return the calculated end effector position
            #coords =tm(bottomTb @ fsr.TAAtoTM(a.reshape((6,1))))
            coords = bottomTb @ tm(a)# @ tm([0, 0, self.tThickness, 0, 0, 0])

            #Disabling these cause unknown issues so far.
            #self._bottomT = bottomTb
            #self._topT = coords


            self.IKHelper(bottomTb, coords, protect = True)
            self._bottomT = bottomTb #@ tm([0, 0, self.bThickness, 0, 0, 0])
            self._topT = coords #@ tm([0, 0, self.tThickness, 0, 0, 0])
            if self.debug:
                disp("Returning from Raphson FK")
            return bottomTb, tm(coords)
        except Exception as e:

            if self.debug:
                disp("Raphson FK Failed due to: " + str(e))
            self.failcount+=1
            return self.FKSciRaphson(L, bottomTb, reverse, protect)


    def LambdaRTP(self, stopt):
        """
        Only used as an assistance function for fixing plate alignment
        """
        lf1 = fsr.LocalToGlobal(stopt, self.reorients[0])
        lf2 = fsr.LocalToGlobal(stopt, self.reorients[1])
        lf3 = fsr.LocalToGlobal(stopt, self.reorients[2])

        d1 = fsr.Distance(lf1, tm([self._tJS[0,0], self._tJS[1,0], self._tJS[2,0], 0,0,0]))
        d2 = fsr.Distance(lf2, tm([self._tJS[0,2], self._tJS[1,2], self._tJS[2,2], 0,0,0]))
        d3 = fsr.Distance(lf3, tm([self._tJS[0,4], self._tJS[1,4], self._tJS[2,4], 0,0,0]))
        return np.array([d1 , d2 , d3])

    def ReorientTopPlate(self):
        """
        Subfunction of fixUpsideDown, responsible for orienting the top plate transform after mirroring
        """
        tt = self.gtopT() @ tm([0, 0, -self.tThickness, 0, 0, 0])
        res = lambda x : self.LambdaRTP(tm([tt[0], tt[1], tt[2], x[0], x[1], x[2]]))
        x0 = self.gtopT()[3:6].flatten()
        xs = sci.optimize.fsolve(res, x0)
        tt[3:6] = xs
        self._topT = tt @ tm([0, 0, self.tThickness, 0, 0, 0])
        #disp(self.LambdaRTP(self.gtopT() @ tm([0, 0, -self.tThickness, 0, 0, 0])))


    def fixUpsideDown(self):
        """
        In situations where the top plate is inverted underneath the bottom plate, yet lengths are valid,
        This function can be used to mirror all the joint locations and "fix" the resultant problem
        """
        for num in range(6):
            #reversable = fsr.GlobalToLocal(tm([self._tJS[0,num], self._tJS[1,num],self._tJS[2,num],0,0,0]), tm([self._bJS[0,num], self._bJS[1,num],self._bJS[2,num],0,0,0]))
            #newTJ = tm([self._bJS[0,num], self._bJS[1,num],self._bJS[2,num],0,0,0]) @ reversable
            newTJ = fsr.Mirror(self.gbottomT() @ tm([0, 0, -self.bThickness, 0, 0, 0]), tm([self._tJS[0,num], self._tJS[1,num],self._tJS[2,num],0,0,0]))
            self._tJS[0, num] = newTJ[0]
            self._tJS[1, num] = newTJ[1]
            self._tJS[2, num] = newTJ[2]
            self.lengths[num] = fsr.Distance(self._tJS[:, num], self._bJS[:, num])
        tt = fsr.Mirror(self.gbottomT() @ tm([0, 0, -self.bThickness, 0, 0, 0]), self.gtopT() @ tm([0, 0, -self.tThickness, 0, 0, 0]))
        tt[3:6] = self.gtopT()[3:6] * -1
        self._topT = tt @ tm([0, 0, self.tThickness, 0, 0, 0])
        self.ReorientTopPlate()

    def validateLegs(self, valid = True, donothing = False):
        if self.validation_settings[0]:
            tval = self._legLengthConstraint(donothing)
            valid = valid and tval
            if not tval:
                self.validation_error += "Leg Length Constraint Violated "
            if not tval and not donothing:
                if self.debug:
                    disp("Executing Length Corrective Action...")
                self._lengthCorrectiveAction()
                valid = self.validate(True, 1)
        return valid

    def validateContinuousTranslation(self, valid=True, donothing = False):
        if self.validation_settings[1]:
            tval = self._continuousTranslationConstraint()
            valid = valid and tval
            if not tval:
                self.validation_error += "Platform Inversion Constraint Violated "
            if not tval and not donothing:
                if self.debug:
                    disp("Executing Continuous Translation Corrective Action...")
                self._continuousTranslationCorrectiveAction()
                valid = self.validate(True, 2)
        return valid
    def validateInteriorAngles(self, valid = True, donothing = False):
        if self.validation_settings[2]:
            tval = self._interiorAnglesConstraint()
            valid = valid and tval
            if not tval:
                self.validation_error += "Interior Angles Constraint Violated "
            if not tval and not donothing:
                if self.debug:
                    disp("Executing Interior Angles Corrective Action...")
                self.IK(bottomT = self.gbottomT(), topT = self.gbottomT() @ self.poffmat, protect = True)
                valid = self.validate(True, 3)
        return valid

    def validatePlateRotation(self, valid = True, donothing = False):
        if self.validation_settings[3]:
            tval = self._plateRotationConstraint()
            valid = valid and tval
            if not tval:
                self.validation_error += "Plate Tilt/Rotate Constraint Violated "
            if not tval and not donothing:
                if self.debug:
                    disp("Executing Plate Rotation Corrective Action By Resetting Platform")
                #disp(self.poffmat)
                self.IK(bottomT = self.gbottomT(), topT = self.gbottomT() @ self.poffmat, protect = True)
                valid = self.validate(True, 4)
        return valid

    def validate(self, donothing = False, topend = 4):
        """
        Validate the position of the stewart platform
        """
        valid = True #innocent until proven INVALID
        #if self.debug:
        #    disp("Validating")
        #First check to make sure leg lengths are not exceeding limit points
        if fsr.Distance(self.gtopT(), self.gbottomT()) > 2 * self.poffset:
            valid = False

        if topend > 0: valid = self.validateLegs(valid, donothing)
        if topend > 1: valid = self.validateContinuousTranslation(valid, donothing)
        if topend > 2: valid = self.validateInteriorAngles(valid, donothing)
        if topend > 3: valid = self.validatePlateRotation(valid, donothing)

        if valid:
            self.validation_error = ""

        return valid

    def _plateRotationConstraint(self):
        valid = True
        for i in range(3):
            if self.curTrans.gTM()[i,i] <= self.plateRotationLimit - .0001:
                if self.debug:
                    disp(self.curTrans.gTM(), "Erroneous TM")
                    print([self.curTrans.gTM()[i,i], self.plateRotationLimit])
                valid = False
        return valid

    def _legLengthConstraint(self, donothing):
        """
        Evaluate Leg Length Limitations of Stewart Platform
        """
        valid = True
        if(np.any(self.lengths < self.lmin) or np.any(self.lengths > self.lmax)):
            valid = False
        return valid

    def _resclLegs(self,cMin,cMax):

        for i in range(6):
            self.lengths[i] = (self.lengths[i]-cMin)/(cMax-cMin) * (min(self.lmax,cMax)-max(self.lmin,cMin)) + max(self.lmin,cMin)

    def _addLegs(self, cMin, cMax):
        boostamt = ((self.lmin-cMin)+self.legSafety)
        if self.debug:
            print("Boost Amount: " + str(boostamt))
        self.lengths += boostamt

    def _subLegs(self, cMin, cMax):
        #print([cMax, self.lmax, cMin, self.lmin, cMax - (cMax - self.lmax + self.legSafety)])
        self.lengths -= ((cMax - self.lmax)+self.legSafety)
        #print(self.lengths)
    def _lengthCorrectiveAction(self):
        """
        Make an attempt to correct leg lengths that are out of bounds.
        Will frequently result in a home-like position
        """
        if self.debug:
            disp(self.lengths, "Lengths Pre Correction")
            disp(self.lengths[np.where(self.lengths > self.lmax)], "over max")
            disp(self.lengths[np.where(self.lengths < self.lmin)], "below min")

        cMin = min(self.lengths.flatten())
        cMax = max(self.lengths.flatten())

        #for i in range(6):
        #    self.lengths[i] = (self.lengths[i]-cMin)/(cMax-cMin) * (min(self.lmax,cMax)-max(self.lmin,cMin)) + max(self.lmin,cMin)
        if cMin < self.lmin and cMax > self.lmax:
            self._resclLegs(cMin,cMax)
            self.validation_error+= " CMethod: Rescale, "
        elif cMin < self.lmin and cMax + (self.lmin - cMin) + self.legSafety < self.lmax:
            self._addLegs(cMin,cMax)
            self.validation_error+= " CMethod: Boost, "
        elif cMax > self.lmax and cMin - (cMax - self.lmax) - self.legSafety > self.lmin:
            self.validation_error+= " CMethod: Subract, "
            self._subLegs(cMin,cMax)
        else:
            self._resclLegs(cMin,cMax)
            self.validation_error+= " CMethod: Unknown Rescale, "

        #self.lengths[np.where(self.lengths > self.lmax)] = self.lmax
        #self.lengths[np.where(self.lengths < self.lmin)] = self.lmin
        if self.debug:
            disp(self.lengths, "Corrected Lengths")
        #disp("HEre's what happened")
        self.FK(self.lengths.copy(), protect = True)
        #print(self.lengths)

    def _continuousTranslationConstraint(self):
        """
        Ensure that the plate is above the prior
        """
        valid = True
        bot = self.gbottomT()
        for i in range(6):
            if fsr.GlobalToLocal(self.gbottomT(), self.gtopT())[2] < 0:
                valid = False
        return valid

    def _continuousTranslationCorrectiveAction(self):
        """
        I'm not sure how to adequately fix that except for mirroring,
        so we'll just return to home instead
        """
        self.IK(topT = self.gbottomT() @ self.poffmat, protect = True)

    def _interiorAnglesConstraint(self):
        """
        Ensures no invalid internal angles
        """
        angles = abs(self.AngleFromNorm())
        if(np.any(np.isnan(angles))):
            return False
        if(np.any(angles > self.jointDeflectionMax)):
            return False
        return True

    def AngleFromNorm(self):
        """
        Returns the angular deviation of each angle socket from its nominal position in radians
        """
        deltAnglesB = np.zeros((6))
        deltAnglesA = np.zeros((6))
        botRot = self.gbottomT()
        topRot = self.gtopT()
        for i in range(6):

                top_joint_i = tm([self._tJS.T[i][0],self._tJS.T[i][1],self._tJS.T[i][2],topRot[3],topRot[4],topRot[5]])
                bottom_joint_i = tm([self._bJS.T[i][0],self._bJS.T[i][1],self._bJS.T[i][2],botRot[3],botRot[4],botRot[5]])

                #We have the relative positions to the top plate of the bottom joints (bottom angles) in home pose
                #We have the relative positions to the bottom plate of the top joints (baseAngles) in home pose
                bottom_to_top_local_home = self.baseAngles[i].copy()
                top_to_bottom_local_home = self.bottomAngles[i].copy()

                #We acquire the current relative (local positions of each)
                bottom_to_top_local = fsr.GlobalToLocal(self.gbottomT(), top_joint_i)
                top_to_bottom_local = fsr.GlobalToLocal(self.gtopT(), bottom_joint_i)

                #We acquire the base positions of each joint
                bottom_to_bottom_local = fsr.GlobalToLocal(self.gbottomT(), bottom_joint_i)
                top_to_top_local = fsr.GlobalToLocal(self.gtopT(), top_joint_i)

                deltAnglesA[i] = fsr.AngleBetween(bottom_to_top_local, bottom_to_bottom_local, bottom_to_top_local_home)
                deltAnglesB[i] = fsr.AngleBetween(top_to_bottom_local, top_to_top_local, top_to_bottom_local_home)

            #DeltAnglesA are the Angles From Norm Bottom
            #DeltAnglesB are the Angles from Norm TOp
        return np.hstack((deltAnglesA, deltAnglesB))

    def AngleFromVertical(self):
        topdown = np.zeros((6))
        bottomup = np.zeros((6))
        for i in range(6):
            tb = self._tJS[:,i].copy().flatten()
            tb[2] = 0
            tt = self._bJS[:,i].copy().flatten()
            tt[2] = tt[2] + 1
            ang = fsr.AngleBetween(self._bJS[:,i], self._tJS[:,i], tb)
            angup = fsr.AngleBetween(self._tJS[:,i], self._bJS[:,i], tt)
            topdown[i] = ang
            bottomup[i] = angup
        return topdown, bottomup

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
    def componentForces(self, tau):
        verts = np.zeros((6))
        horzs = np.zeros((6))
        for i in range(6):
            tb = self._tJS[:,i].copy().flatten()
            tb[2] = 0
            ang = fsr.AngleBetween(self._bJS[:,i], self._tJS[:,i], tb)
            vertf = tau[i] * np.sin(ang)
            horzf = tau[i] * np.cos(ang)
            verts[i] = vertf
            horzs[i] = horzf
        return verts, horzs

    def _bottomTopCheck(self, bottomT, topT):
        """
        Checks to make sure that a bottom and top provided are not null
        """
        if bottomT == None:
            bottomT = self.gbottomT()
        if topT == None:
            topT = self.gtopT()
        return bottomT, topT

    def JacobianSpace(self, bottomT = None, topT = None):
        """
        Calculates space jacobian for stewart platform. Takes in bottom transform and top transform
        """
        #If not supplied paramters, draw from stored values
        bottomT, topT = self._bottomTopCheck(bottomT, topT)
        #Just invert the inverted
        InvJac = self.InverseJacobianSpace(bottomT, topT)
        return ling.pinv(InvJac)


    def InverseJacobianSpace(self, bottomT = None, topT = None, protect = True):
        """
        Calculates Inverse Jacobian for stewart platform. Takes in bottom and top transforms
        """
        #Ensure everything is kosher with the plates
        bottomT, topT = self._bottomTopCheck(bottomT, topT)

        #Store old values
        oldbt = self.gbottomT()
        oldtT = self.gtopT()

        #Perform IK on bottom and top
        self.IK(bottomT, topT, protect = protect)

        #Create Jacobian
        JsinvT = np.zeros((6,6))
        for i in range(6):
            #todo check sign on nim,
            ni = fmr.Normalize(self._tJS[:,i]-self._bJS[:,i]) #Reverse for upward forces?
            qi = self._bJS[:,i]
            col = np.hstack((np.cross(qi,ni),ni))
            JsinvT[:,i] = col
        Jsinv = JsinvT.T

        #Restore original Values
        self.IK(oldbt,oldtT, protect = protect)
        return Jsinv

    #Returns Top Down Jacobian instead of Bottom Up
    def AltInverseJacobianSpace(self, bottomT = None, topT = None, protect = True):
        """
        Returns top down jacobian instead of bottom up
        """
        bottomT, topT = self._bottomTopCheck(bottomT, topT)
        oldbt = copy.copy(bottomT)
        oldtT = copy.copy(topT)
        self.IK(bottomT, topT)
        JsinvT = np.zeros((6,6))
        for i in range(6):
            ni = fmr.Normalize(self._bJS[:,i]-self._tJS[:,i])
            qi = self._tJS[:,i]
            JsinvT[:,i] = np.hstack((np.cross(qi,ni),ni))
        Jsinv = JsinvT.conj().transpose()

        self.IKHelper(oldbt,oldtT)

        return Jsinv

    #Adds in actuator and plate forces, useful for finding forces on a full stack assembler
    def CarryMassCalc(self, twrench, protect = False):
        """
        Calculates the forces on each leg given their masses, masses of plates, and a wrench on the end effector. Returns a wrench on the bottom plate
        """
        wrench = twrench.copy()
        wrench = wrench + fsr.GenForceWrench(self.gtopT(), self.plateTN, self.dir)
        tau = self.MeasureForcesFromWrenchEE(self.gbottomT(), self.gtopT(), wrench, protect = protect)
        for i in range(6):
            #print(self.getActuatorLoc(i, 't'))
            wrench += fsr.GenForceWrench(self.getActuatorLoc(i, 't'), self.actuatorTN, self.dir)
            wrench += fsr.GenForceWrench(self.getActuatorLoc(i, 'b'), self.actuatorBN, self.dir)
        wrench = wrench + fsr.GenForceWrench(self.gbottomT(), self.plateBN, self.dir)
        return tau, wrench

    def CarryMassCalcNew(self, twrench, protect = False):
        #We will here assume that the wrench is in the local frame of the top platform.
        wrench = twrench.copy()
        wrench = wrench + fsr.GenForceWrench(tm(), self.plateTN, self.dir)
        tau = self.MeasureForcesAtEENew(wrench, protect = protect)
        wrenchCoF = fsr.TransformWrenchFrame(wrench, self.gtopT(), self.gbottomT())

        for i in range(6):
            #print(self.getActuatorLoc(i, 't'))
            #The following representations are equivalent.
            wrenchCoF += fsr.GenForceWrench(fsr.GlobalToLocal(self.getActuatorLoc(i, 't'), self.gbottomT()), self.actuatorTN, self.dir)
            wrenchCoF += fsr.GenForceWrench(fsr.GlobalToLocal(self.getActuatorLoc(i, 'b'), self.gbottomT()), self.actuatorBN, self.dir)
            #wrenchCoF += fsr.TransformWrenchFrame(fsr.GenForceWrench(tm(), self.actuatorTN, self.dir), self.getActuatorLoc(i, 't'), self.gbottomT())
            #wrenchCoF += fsr.TransformWrenchFrame(fsr.GenForceWrench(tm(), self.actuatorBN, self.dir), self.getActuatorLoc(i, 'b'), self.gbottomT())
        wrenchCoF = wrenchCoF + fsr.GenForceWrench(tm(), self.plateBN, self.dir)
        return tau, wrenchCoF

    def MeasureForcesAtEENew(self, wrench, protect = False):
        Js = ling.pinv(self.InverseJacobianSpace(self.gbottomT(), self.gtopT(), protect = protect))
        tau = Js.T @ wrench
        self.legForces = tau
        return tau

    def CarryMassCalcUp(self, twrench, protect = False):
        wrench = twrench.copy()
        wrench = wrench + fsr.GenForceWrench(self.gbottomT(), self.bPlateMass * self.grav, np.array([0, 0, -1]))
        tau = self.MeasureForcesFromBottomEE(self.gbottomT(), self.gtopT(), wrench, protect = protect)
        for i in range(6):
            wrench += fsr.GenForceWrench(self.getActuatorLoc(i, 't'), self.actuatorTopMass * self.grav, np.array([0, 0, -1]))
            wrench += fsr.GenForceWrench(self.getActuatorLoc(i, 'b'), self.actuatorBottomMass * self.grav, np.array([0, 0, -1]))
        wrench = wrench + fsr.GenForceWrench(self.gtopT(), self.tPlateMass * self.grav, np.array([0, 0, -1]))
        return tau, wrench

    #Get Force wrench from the End Effector Force
    def MeasureForcesFromWrenchEE(self, bottomT = np.zeros((1)) , topT = np.zeros((1)), topWEE = np.zeros((1)), protect = True):
        """
        Calculates forces on legs given end effector wrench
        """
        bottomT, topT = self._bottomTopCheck(bottomT, topT)
        if topWEE.size < 6:
            disp("Please Enter a Wrench")
        #topWS = fmr.Adjoint(ling.inv(topT)).conj().transpose() @ topWEE
        #Modern Robotics 3.95 Fb = Ad(Tba)^T * Fa
        #topWS = topT.inv().Adjoint().T @ topWEE
        topWS = fsr.TransformWrenchFrame(topWEE, tm(), topT)
        Js = ling.pinv(self.InverseJacobianSpace(bottomT, topT, protect = protect))
        tau = Js.T @ topWS
        self.legForces = tau
        return tau

    def MeasureForcesFromBottomEE(self, bottomT = np.zeros((1)) , topT = np.zeros((1)), topWEE = np.zeros((1)), protect = True):
        """
        Calculates forces on legs given end effector wrench
        """
        bottomT, topT = self._bttomTopCheck(bottomT, topT)
        if topWEE.size < 6:
            disp("Please Enter a Wrench")
        #topWS = fmr.Adjoint(ling.inv(topT)).conj().transpose() @ topWEE
        bottomWS = bottomT.inv().Adjoint().T @ topWEE
        Js = ling.pinv(self.InverseJacobianSpace(bottomT, topT, protect = protect))
        tau = Js.T @ bottomWS
        self.legForces = tau
        return tau

    def WrenchEEFromMeasuredForces(self,bottomT,topT,tau):
        """
        Calculates wrench on end effector from leg forces
        """
        self.legForces = tau
        Js = ling.pinv(self.InverseJacobianSpace(bottomT,topT))
        topWS = ling.inv(Js.conj().transpose()) @ tau
        #self.topWEE = fmr.Adjoint(topT).conj().transpose() @ topWS
        self.topWEE = topT.Adjoint().conj().transpose() @ topWS
        return self.topWEE, topWS, Js

    def WrenchBottomFromMeasuredForces(self, bottomT, topT, tau):
        """
        Unused. Calculates wrench on the bottom plate from leg forces
        """
        self.legForces = tau
        Js = ling.pinv(self.AltInverseJacobianSpace(bottomT, topT))
        bottomWS = ling.inv(Js.conj().transpose()) @ tau
        #self.bottomWEE = fmr.Adjoint(bottomT).conj().transpose() @ bottomWS
        self.bottomWEE = bottomT.Adjoint().conj().transpose() @ bottomWS
        return self.bottomWEE, bottomWS, Js

    def SumActuatorWrenches(self, forces = None):
        if forces is None:
            forces = self.legForces

        wrench = fsr.GenForceWrench(tm(), 0, [0,0,-1])
        for i in range(6):
            unitvec = fmr.Normalize(self._bJS[:,i]-self._tJS[:,i])
            wrench += fsr.GenForceWrench(self._tJS[:,i], float(forces[i]), unitvec)
        #wrench = fsr.TransformWrenchFrame(wrench, tm(), self.gtopT())
        return wrench


    def move(self, T, protect = False):
        """
        Move entire Assembler Stack to another location and orientation
        This function and syntax are shared between all kinematic structures.
        """
        #Moves the base of the stewart platform to a new location


        self.curTrans = fsr.GlobalToLocal(self.gbottomT(), self.gtopT())
        self._bottomT = T.copy()
        self.IK(topT = fsr.LocalToGlobal(self.gbottomT(), self.curTrans), protect = protect)


def LoadSP(fname, file_directory = "../RobotDefinitions/", baseloc = None, altRot = 1):
    print(fname)
    print(file_directory)
    total_name = file_directory + fname
    print(total_name)
    with open(total_name, "r") as sp_file:
        sp_data = json.load(sp_file)
    bot_radius = sp_data["BottomPlate"]["JointRadius"] #Radius of Ball Joint Circle in Meters
    top_radius = sp_data["TopPlate"]["JointRadius"]
    bot_joint_spacing = sp_data["BottomPlate"]["JointSpacing"] #Spacing in Degrees
    top_joint_spacing = sp_data["TopPlate"]["JointSpacing"]
    bot_thickness = sp_data["BottomPlate"]["Thickness"]
    top_thickness = sp_data["TopPlate"]["Thickness"]
    OuterTopRad = sp_data["Drawing"]["TopRadius"]
    OuterBotRad = sp_data["Drawing"]["BottomRadius"]
    ShaftRad = sp_data["Drawing"]["ShaftRadius"]
    MotorRad = sp_data["Drawing"]["MotorRadius"]
    actuator_shaft_mass = 0
    actuator_motor_mass = 0
    plate_top_mass = 0
    plate_bot_mass = 0
    motor_grav_center = 0
    shaft_grav_center = 0
    name = sp_data["Name"]
    actuator_min = sp_data["Actuators"]["MinExtension"] #meters
    actuator_max = sp_data["Actuators"]["MaxExtension"]
    force_lim = sp_data["Actuators"]["ForceLimit"]
    max_dev = sp_data["Settings"]["MaxAngleDev"]
    if sp_data["Settings"]["AssignMasses"] == 1:
        actuator_motor_mass = sp_data["Actuators"]["MotorMass"]
        actuator_shaft_mass = sp_data["Actuators"]["ShaftMass"]
        plate_top_mass = sp_data["TopPlate"]["Mass"]
        plate_bot_mass = sp_data["BottomPlate"]["Mass"]
        if sp_data["Settings"]["InferActuatorCOG"] == 1:
            motor_grav_center = sp_data["Actuators"]["MotorCOGD"]
            shaft_grav_center = sp_data["Actuators"]["ShaftCOGD"]
        else:
            inferred_cog = 1/4 * (actuator_min+actuator_max)/2
            actuator_shaft_mass = inferred_cog
            motor_grav_center = inferred_cog
    if baseloc == None:
        baseloc = tm()


    newsp = NewSP(bot_radius, top_radius, bot_joint_spacing, top_joint_spacing,
        bot_thickness, top_thickness, actuator_shaft_mass, actuator_motor_mass, plate_top_mass,
        plate_bot_mass, motor_grav_center, shaft_grav_center, actuator_min, actuator_max, baseloc, name, altRot)

    newsp.SetDrawingDimensions(OuterTopRad, OuterBotRad, ShaftRad, MotorRad)
    newsp.SetAngleDev(max_dev)
    newsp.force_limit = force_lim

    return newsp
def NewSP(bRadius, tRadius, bJointSpace, tJointSpace, bThickness, tThickness, actuator_shaft_mass,
    actuator_motor_mass, plate_top_mass, plate_bot_mass, motor_grav_center, shaft_grav_center,
    actuator_min, actuator_max, baseLoc, name, rot = 1):

    bGapS = bJointSpace / 2 * np.pi / 180
    tGapS = tJointSpace / 2 * np.pi / 180

    gapL = 120 * np.pi / 180 #Angle of seperation between joint clusters
    gapH = 60 * np.pi / 180 #Offset in rotation of the top plate versus the bottom plate

    bangles =  np.array([-bGapS, bGapS, gapL-bGapS, gapL+bGapS, 2*gapL-bGapS, 2*gapL+bGapS])
    tangles = np.array([-gapH+tGapS, gapH-tGapS, gapH+tGapS, gapH+gapL-tGapS, gapH+gapL+tGapS, -gapH-tGapS])
    if rot == -1:
        tangles =  np.array([-bGapS, bGapS, gapL-bGapS, gapL+bGapS, 2*gapL-bGapS, 2*gapL+bGapS])
        bangles = np.array([-gapH+tGapS, gapH-tGapS, gapH+tGapS, gapH+gapL-tGapS, gapH+gapL+tGapS, -gapH-tGapS])

    S = fmr.ScrewToAxis(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 0).reshape((6,1))

    Mb = tm(np.array([bRadius, 0.0, 0.0, 0.0, 0.0, 0.0])) #how far from the bottom plate origin should clusters be generated
    Mt = tm(np.array([tRadius, 0.0, 0.0, 0.0, 0.0, 0.0])) #Same thing for the top

    bj = np.zeros((3, 6)) #Pre allocate arrays
    tj = np.zeros((3, 6))

    for i in range(0,6):
        bji = fsr.TransformFromTwist(bangles[i] * S) @ Mb
        tji = fsr.TransformFromTwist(tangles[i] * S) @ Mt
        bj[0:3,i] = bji[0:3].reshape((3))
        tj[0:3,i] = tji[0:3].reshape((3))
        bj[2,i] = bThickness
        tj[2,i] = -tThickness

    bottom = baseLoc.copy()
    tentative_height = MidHeightEstimate(actuator_min, actuator_max, bj, bThickness, tThickness)
    if rot == -1:
        tentative_height = MidHeightEstimate(actuator_min, actuator_max, tj, bThickness, tThickness)
    top = bottom @ tm(np.array([0.0, 0.0, tentative_height, 0.0, 0.0, 0.0]))

    newsp = SP(bj, tj, bottom, top, actuator_min, actuator_max, bThickness, tThickness, name)
    newsp.SetMasses(plate_bot_mass, actuator_shaft_mass, actuator_motor_mass, tPlateMass = plate_top_mass)
    newsp.SetCOG(motor_grav_center, shaft_grav_center)

    return newsp
def MakeSP(bRad, tRad, spacing, baseT, platOffset, rot = -1, plateThickness = 0, lset = None, altRot = 0):
    """
    Builds a new stewart platform object.
    Takes in Bottom Radius, Top Radius, Spacing between joint clusters in degrees, base transform,
    height of the top platform in neutral pose, and rotational offset
    """
    gapS = spacing/2*np.pi/180 #Angle between cluster joints
    gapL = 120*np.pi/180 #Angle of seperation between joint clusters
    gapH = 60*np.pi/180 #Offset in rotation of the top plate versus the bottom plate
    bangles = np.array([-gapS, gapS, gapL-gapS, gapL+gapS, 2*gapL-gapS, 2*gapL+gapS]) + altRot*np.pi/180
    tangles = np.array([-gapH+gapS, gapH-gapS, gapH+gapS, gapH+gapL-gapS, gapH+gapL+gapS, -gapH-gapS])+ altRot*np.pi/180
    if rot == -1:
        tangles = np.array([-gapS, gapS, gapL-gapS, gapL+gapS, 2*gapL-gapS, 2*gapL+gapS])+ altRot*np.pi/180
        bangles = np.array([-gapH+gapS, gapH-gapS, gapH+gapS, gapH+gapL-gapS, gapH+gapL+gapS, -gapH-gapS])+ altRot*np.pi/180

    disp(bangles, "bangles")
    disp(tangles, "tangles")
    S = fmr.ScrewToAxis(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 0).reshape((6,1))

    Mb = tm(np.array([bRad, 0.0, 0.0, 0.0, 0.0, 0.0])) #how far from the bottom plate origin should clusters be generated
    Mt = tm(np.array([tRad, 0.0, 0.0, 0.0, 0.0, 0.0])) #Same thing for the top

    bj = np.zeros((3, 6)) #Pre allocate arrays
    tj = np.zeros((3, 6))

    #Generate position vectors (XYZ) for top and bottom joint locations
    for i in range(0,6):
        bji = fsr.TransformFromTwist(bangles[i] * S) @ Mb
        tji = fsr.TransformFromTwist(tangles[i] * S) @ Mt
        bj[0:3,i] = bji[0:3].reshape((3))
        tj[0:3,i] = tji[0:3].reshape((3))
        bj[2,i] = plateThickness/2
        tj[2,i] = -plateThickness/2

    #if rot == -1:
    #    disp(bj, "Prechange")
#
#        rotby = TAAtoTM(np.array([0,0,0,0,0,np.pi/3]))
#        for i in range(6):
#            bj[0:3,i] = TMtoTAA(rotby @ TAAtoTM(np.array([bj[0,i],bj[1,i],bj[2,i],0,0,0])))[0:3].reshape((3))
#            tj[0:3,i] = TMtoTAA(rotby @ TAAtoTM(np.array([tj[0,i],tj[1,i],tj[2,i],0,0,0])))[0:3].reshape((3))
#        disp(bj, "postchange")
    bottom = baseT.copy()
    #Generate top position at offset from the bottom position
    top = bottom @ tm(np.array([0.0, 0.0, platOffset, 0.0, 0.0, 0.0]))
    sp = SP(bj, tj, bottom, top, 0, 0, plateThickness, plateThickness, 'sp')
    sp.bRad = bRad
    sp.tRad = tRad

    return sp, bottom, top
#Helpers
def MidHeightEstimate(lmin, lmax, bj, bth, tth):
    s1 = (lmin + lmax) / 2
    d1 = fsr.Distance(tm([bj[0,0], bj[1,0], bj[2,0], 0, 0, 0]),
            tm([bj[0,1], bj[1,1], bj[2,1], 0, 0, 0]))
    hest = (np.sqrt(s1 ** 2 - d1 **2)) + bth + tth
    return hest
