from faser_math import tm, fmr, fsr
from faser_utils.disp.disp import disp
import numpy as np
import scipy as sci
import scipy.linalg as ling
import copy
import json

class SP:
    #Conventions:
    #Filenames:  snake_case
    #Variables: snake_case
    #Functions: camelCase
    #ClassNames: CapsCase
    #Docstring: Google
    def __init__(self, bottom_joints, top_joints, bT, tT, leg_ext_min,
        leg_ext_max, bottom_plate_thickness, top_plate_thickness, name):
        """
        Initializes a new Stewart Platform Object

        Args:
            bottom_joints (ndarray): Bottom joint positions of the stewart platform
            top_joints (ndarray): Top joint positions of the stewart platform
            bT (tm): bottom plate position
            tT (tm): top plate position
            leg_ext_min (float): minimum leg ext limit
            leg_ext_max (float): maximum leg ext limit
            bottom_plate_thickness (float): bottom plate thickness
            top_plate_thickness (float): top plate thickness
            name (string): name of the sp
        Returns:
            SP: sp model object

        """
        self.bottom_joints = np.copy(bottom_joints)
        self.top_joints = np.copy(top_joints)
        self.bottom_joints_init = self.bottom_joints.conj().transpose()
        self.top_joints_init = self.top_joints.conj().transpose()
        self.bottom_plate_pos = bT.copy()
        self.top_plate_pos = tT.copy()
        self.bottom_joints_space = np.zeros((3, 6))
        self.top_joints_space = np.zeros((3, 6))

        #Debug
        self.leg_ext_safety = .001
        self.debug = 0

        #Physical Parameters
        self.bottom_plate_thickness = bottom_plate_thickness
        self.top_plate_thickness = top_plate_thickness
        if leg_ext_min == 0:
            self.leg_ext_min = 0
            self.leg_ext_max = 2
        self.leg_ext_min = leg_ext_min
        self.leg_ext_max = leg_ext_max

        #Reserve Val
        self.nominal_height = fsr.Distance(bT, tT)
        self.nominal_plate_transform = tm([0, 0, self.nominal_height, 0, 0, 0])

        #Drawing Characteristics
        self.outer_top_radius = 0
        self.outer_bottom_radius = 0
        self.act_shaft_radius = 0
        self.act_motor_radius = 0

        #Empty array indicates these values haven't been populated yet
        self.leg_forces =  np.zeros(1)
        self.top_plate_wrench =  np.zeros(1)
        self.bottom_plate_wrench =  np.zeros(1)

        #Mass values from bottom mass, top mass, and actuator portion masses can be set directly.
        self.bottom_plate_mass = 0
        self.top_plate_mass = 0
        self.act_shaft_mass = 0
        self.act_motor_mass = 0
        self.act_shaft_newton_force = 0
        self.act_motor_newton_force = 0
        self.top_plate_newton_force = 0
        self.bottom_plate_newton_force = 0
        self.grav = 9.81
        self.dir = np.array([0, 0, -1])
        self.act_shaft_grav_center = 0
        self.act_motor_grav_center = 0
        self.force_limit= 0

        #Tolerances and Limits
        self.joint_deflection_max = 140/2*np.pi/180#2*np.pi/5
        self.plate_rotation_limit = np.cos(60*np.pi/180)

        #Newton Settings
        self.tol_f = 1e-5/2
        self.tol_a = 1e-5/2
        self.max_iterations = 1e4

        #Errors and Counts
        self.fail_count = 0
        self.validation_settings = [1, 0, 0, 1]
        self.fk_mode = 1
        self.validation_error = ""

        self.IK(bT, tT, protect = True)

        self.bottom_joint_angles_init = self.top_joints_space.T.copy()
        self.bottom_joint_angles = self.bottom_joints_space.T.copy()

        self.bottom_joint_angles_init = [None] * 6
        self.bottom_joint_angles = [None] * 6
        for i in range(6):
            self.bottom_joint_angles_init[i] = fsr.GlobalToLocal(self.getBottomT(),
                tm([self.top_joints_space.T[i][0], self.top_joints_space.T[i][1],
                self.top_joints_space.T[i][2], 0, 0, 0]))
            self.bottom_joint_angles[i] = fsr.GlobalToLocal(self.getTopT(),
                tm([self.bottom_joints_space.T[i][0], self.bottom_joints_space.T[i][1],
                self.bottom_joints_space.T[i][2], 0, 0, 0]))

        t1 = fsr.GlobalToLocal(self.getTopT() @ tm([0, 0, -self.top_plate_thickness, 0, 0, 0]),
            tm([self.top_joints_space[0, 0],
            self.top_joints_space[1, 0],
            self.top_joints_space[2, 0], 0, 0, 0]))
        t2 = fsr.GlobalToLocal(self.getTopT() @ tm([0, 0, -self.top_plate_thickness, 0, 0, 0]),
            tm([self.top_joints_space[0, 2],
            self.top_joints_space[1, 2],
            self.top_joints_space[2, 2], 0, 0, 0]))
        t3 = fsr.GlobalToLocal(self.getTopT() @ tm([0, 0, -self.top_plate_thickness, 0, 0, 0]),
            tm([self.top_joints_space[0, 4],
            self.top_joints_space[1, 4],
            self.top_joints_space[2, 4], 0, 0, 0]))
        self.reorients = [t1, t2, t3]


        #Compatibility
        self.plate_thickness_avg = (self.top_plate_thickness + self.bottom_plate_thickness) / 2
        self.nominal_plate_transform = tm([0, 0, self.plate_thickness_avg, 0, 0, 0])

        #Validation Settings


    """
       _____      _   _                                     _    _____      _   _
      / ____|    | | | |                    /\             | |  / ____|    | | | |
     | |  __  ___| |_| |_ ___ _ __ ___     /  \   _ __   __| | | (___   ___| |_| |_ ___ _ __ ___
     | | |_ |/ _ \ __| __/ _ \ '__/ __|   / /\ \ | '_ \ / _` |  \___ \ / _ \ __| __/ _ \ '__/ __|
     | |__| |  __/ |_| ||  __/ |  \__ \  / ____ \| | | | (_| |  ____) |  __/ |_| ||  __/ |  \__ \
      \_____|\___|\__|\__\___|_|  |___/ /_/    \_\_| |_|\__,_| |_____/ \___|\__|\__\___|_|  |___/

    """
    def setMasses(self, plate_mass_general, act_shaft_mass,
        act_motor_mass, grav = 9.81, top_plate_mass = 0):
        """
        Set masses for each SP in the Assembler, note that because central platforms
        share plates, these weights are halved with respect to end plates
        Args:
            plate_mass_general (float): mass of bottom plate (both if top is not specified) (kg)
            act_shaft_mass (float): mass of actuator shaft (kg)
            act_motor_mass (float): mass of actuator motor (kg)
            grav (float):  [Optional, default 9.81] acceleration due to gravity
            top_plate_mass (float): [Optional, default 0] top plate mass (kg)

        Returns:
            type: Description of returned object.

        """
        self.bottom_plate_mass = plate_mass_general
        if top_plate_mass != 0:
            self.top_plate_mass = top_plate_mass
        else:
            self.top_plate_mass = plate_mass_general
        self.setGrav(grav)
        self.act_shaft_mass = act_shaft_mass
        self.act_motor_mass = act_motor_mass
        self.act_motor_newton_force = self.act_motor_mass * self.grav
        self.act_shaft_newton_force = self.act_shaft_mass * self.grav
        self.top_plate_newton_force = self.top_plate_mass * self.grav
        self.bottom_plate_newton_force = self.bottom_plate_mass * self.grav

    def setGrav(self, grav = 9.81):
        """
        Sets Gravity
        Args:
            grav (float): Acceleration due to gravity
        Returns:
            None: None
        """
        self.grav = grav

    def setCOG(self, motor_grav_center, shaft_grav_center):
        """
        Sets the centers of gravity for actuator components
        Args:
            motor_grav_center (float): distance from top of actuator to actuator shaft COG
            shaft_grav_center (float): distance from bottom of actuator to actuator motor COG
        """
        self.act_shaft_grav_center = shaft_grav_center
        self.act_motor_grav_center = motor_grav_center

    def setMaxAngleDev(self, max_angle_dev = 55):
        """
        Set the maximum angle joints can deflect before failure
        Args:
            max_angle_dev (float): maximum deflection angle (degrees)
        """
        self.joint_deflection_max = max_angle_dev*np.pi/180

    def setMaxPlateRotation(self, max_plate_rotation = 60):
        """
        Set the maximum angle the plate can rotate before failure

        Args:
            max_plate_rotation (Float): Maximum angle before plate rotation failure (degrees)
        """
        self.plate_rotation_limit = np.cos(max_plate_rotation * np.pi / 180)

    def setDrawingDimensions(self, outer_top_radius,
        outer_bottom_radius, act_shaft_radius, act_motor_radius):
        """

        Args:
            outer_top_radius (Float): Description of parameter `outer_top_radius`.
            outer_bottom_radius (Float): Description of parameter `outer_bottom_radius`.
            act_shaft_radius (Float): Description of parameter `act_shaft_radius`.
            act_motor_radius (Float): Description of parameter `act_motor_radius`.

        Returns:
            type: Description of returned object.

        """
        self.outer_top_radius = outer_top_radius
        self.outer_bottom_radius = outer_bottom_radius
        self.act_shaft_radius = act_shaft_radius
        self.act_motor_radius = act_motor_radius

    def setPlatePos(self, bottom_plate_pos, top_plate_pos):
        """

        Args:
            bottom_plate_pos (tm): Description of parameter `bottom_plate_pos`.
            top_plate_pos (tm): Description of parameter `top_plate_pos`.

        Returns:
            type: Description of returned object.

        """
        if bottom_plate_pos is not None:
            self.bottom_plate_pos = bottom_plate_pos
        if top_plate_pos is not None:
            self.top_plate_pos = top_plate_pos

    def getBottomJoints(self):
        """

        Returns:
            type: Description of returned object.

        """
        return self.bottom_joints_space

    def getTopJoints(self):
        """

        Returns:
            type: Description of returned object.

        """
        return self.top_joints_space

    def getCurrentLocalTransform(self):
        """

        Returns:
            type: Description of returned object.

        """
        return self.current_plate_transform_local

    def getLegForces(self):
        """

        Returns:
            type: Description of returned object.

        """
        return self.leg_forces

    def getLens(self):
        """

        Returns:
            type: Description of returned object.

        """
        """
        returns leg lengths
        """
        return self.lengths.copy()

    def getTopT(self):
        """
        Return the transform of the top plate
        Returns:
            type: Description of returned object.

        """
        return self.top_plate_pos.copy()

    def getBottomT(self):
        """
        Return the transform of the bottom plate

        Returns:
            type: Description of returned object.

        """
        return self.bottom_plate_pos.copy()

    def getActuatorUnitVec(self, point_1, point_2, distance):
        """

        Args:
            point_1 (tm): Description of parameter `point_1`.
            point_2 (tm): Description of parameter `point_2`.
            distance (Float): Description of parameter `distance`.

        Returns:
            type: Description of returned object.

        """
        v1 = np.array([point_1[0], point_1[1], point_1[2]])
        unit_b = (np.array([point_2[0], point_2[1], point_2[2]]) - v1)
        unit = unit_b / ling.norm(unit_b)
        pos = v1 + (unit * distance)
        return tm([pos[0], pos[1], pos[2], 0, 0, 0])

    def getActuatorLoc(self, num, type = 'm'):
        """
        Returns the position of a specified actuator. Takes in an actuator number and a type.
        m for actuator midpoint
        b for actuator motor position
        t for actuator top position

        Args:
            num (Int): Description of parameter `num`.
            type (Char): Description of parameter `type`.

        Returns:
            type: Description of returned object.
        """
        pos = 0
        if type == 'm':
            pos = np.array([(self.bottom_joints_space[0, num] + self.top_joints_space[0, num])/2,
                (self.bottom_joints_space[1, num] + self.top_joints_space[1, num])/2,
                (self.bottom_joints_space[2, num] + self.top_joints_space[2, num])/2])
        bottom_act_joint = tm([self.bottom_joints_space[0, num],
            self.bottom_joints_space[1, num], self.bottom_joints_space[2, num], 0, 0, 0])
        top_act_joint = tm([self.top_joints_space[0, num],
            self.top_joints_space[1, num], self.top_joints_space[2, num], 0, 0, 0])
        if type == 'b':
            #return fsr.TMMidRotAdjust(bottom_act_joint, bottom_act_joint,
            #   top_act_joint, mode = 1) @ tm([0, 0, self.act_motor_grav_center, 0, 0, 0])
            return self.getActuatorUnitVec(bottom_act_joint,
                top_act_joint, self.act_motor_grav_center)
        if type == 't':
            #return fsr.TMMidRotAdjust(top_act_joint, top_act_joint, bottom_act_joint,
            #   mode = 1) @ tm([0, 0, self.act_shaft_grav_center, 0, 0, 0])
            return self.getActuatorUnitVec(top_act_joint,
                bottom_act_joint, self.act_shaft_grav_center)
        new_position = tm([pos[0], pos[1], pos[2], 0, 0, 0])
        return new_position

    def spinCustom(self, rot):
        """

        Args:
            rot (Float): Description of parameter `rot`.

        Returns:
            type: Description of returned object.

        """
        old_base_pos = self.getBottomT()
        self.move(tm())
        current_top_pos = self.getTopT()
        top_joints_copy = self.top_joints_space.copy()
        bottom_joints_copy = self.bottom_joints_space.copy()
        top_joints_origin_copy = self.top_joints[2, 0:6]
        bottom_joints_origin_copy = self.bottom_joints[2, 0:6]
        rotation_transform = tm([0, 0, 0, 0, 0, rot * np.pi / 180])
        self.move(rotation_transform)
        top_joints_space_new = self.top_joints_space.copy()
        bottom_joints_space_new = self.bottom_joints_space.copy()
        top_joints_copy[0:2, 0:6] = top_joints_space_new[0:2, 0:6]
        bottom_joints_copy[0:2, 0:6] = bottom_joints_space_new[0:2, 0:6]
        bottom_joints_copy[2, 0:6] = bottom_joints_origin_copy
        top_joints_copy[2, 0:6] = top_joints_origin_copy
        self.move(tm())
        self.bottom_joints = bottom_joints_copy
        self.top_joints = top_joints_copy
        self.bottom_joints_space = bottom_joints_space_new
        self.top_joints_space = top_joints_space_new
        self.move(old_base_pos)

    def IKPath(self, goal, steps):
        """

        Args:
            goal (tm): Description of parameter `goal`.
            steps (Int): Description of parameter `steps`.

        Returns:
            type: Description of returned object.

        """
        return fsr.IKPath(self.getTopT(), goal, steps)

    def IK(self, bottom_plate_pos = None, top_plate_pos = None, protect = False, dir = 1):
        """

        Args:
            bottom_plate_pos (tm): Description of parameter `bottom_plate_pos`.
            top_plate_pos (tm): Description of parameter `top_plate_pos`.
            protect (Bool): Description of parameter `protect`.
            dir (Int): Description of parameter `dir`.

        Returns:
            type: Description of returned object.

        """

        bottom_plate_pos, top_plate_pos = self.bottomTopCheck(bottom_plate_pos, top_plate_pos)

        leg_lengths, bottom_plate_pos, top_plate_pos = self.IKHelper(
            bottom_plate_pos, top_plate_pos, protect, dir)
        #Determine current transform


        self.bottom_plate_pos = bottom_plate_pos.copy()
        self.top_plate_pos = top_plate_pos.copy()

        #Ensure a valid position
        valid = True
        if not protect:
            valid = self.validate()
        return leg_lengths, valid

    def IKHelper(self, bottom_plate_pos = None, top_plate_pos = None, protect = False, dir = 1):
        """
        Calculates Inverse Kinematics for a single stewart plaform.
        Takes in bottom plate transform, top plate transform, protection paramter, and direction

        Args:
            bottom_plate_pos (tm): Description of parameter `bottom_plate_pos`.
            top_plate_pos (tm): Description of parameter `top_plate_pos`.
            protect (Bool): Description of parameter `protect`.
            dir (Int): Description of parameter `dir`.

        Returns:
            type: Description of returned object.

        """
        #If not supplied paramters, draw from stored values
        bottom_plate_pos, top_plate_pos = self.bottomTopCheck(bottom_plate_pos, top_plate_pos)
        #Check for excessive rotation
        #Poses which would be valid by leg length
        #But would result in singularity
        #Set bottom and top transforms
        #self.bottom_plate_pos = bottom_plate_pos
        #self.top_plate_pos = top_plate_pos

        #Call the IK method from the JIT numba file (FASER HIGH PER)
        #Shoulda just called it HiPer FASER. Darn.
        self.lengths, self.bottom_joints_space, self.top_joints_space = fmr.SPIKinSpace(
            bottom_plate_pos.gTM(), top_plate_pos.gTM(), self.bottom_joints,
            self.top_joints, self.bottom_joints_space, self.top_joints_space)
        self.current_plate_transform_local = fsr.GlobalToLocal(bottom_plate_pos, top_plate_pos)
        return np.copy(self.lengths), bottom_plate_pos, top_plate_pos

    def FK(self, L, bottom_plate_pos =None, reverse = False, protect = False):
        """
        FK Host Function

        Args:
            L (ndarray(Float)): Description of parameter `L`.
            bottom_plate_pos (tm): Description of parameter `bottom_plate_pos`.
            reverse (Bool): Description of parameter `reverse`.
            protect (Bool): Description of parameter `protect`.

        Returns:
            type: Description of returned object.

        """
        #FK host function, calls subfunctions depedning on the value of fk_mode
        #return self.FKSciRaphson(L, bottom_plate_pos, reverse, protect)
        #bottom_plate_pos, n = self._applyPlateTransform(bottom_plate_pos = bottom_plate_pos)
        if self.fk_mode == 0:
            bottom, top = self.FKSolve(L, bottom_plate_pos, reverse, protect)
        else:
            bottom, top = self.FKRaphson(L, bottom_plate_pos, reverse, protect)

        if not self.continuousTranslationConstraint():
            if self.debug:
                disp("FK Resulted In Inverted Plate Alignment. Repairing...")
            #self.IK(top_plate_pos = self.getBottomT() @ tm([0, 0, self.nominal_height, 0, 0, 0]))
            #self.FK(L, protect = True)
            self.fixUpsideDown()
        self.current_plate_transform_local = fsr.GlobalToLocal(bottom, top)
        #self._undoPlateTransform(bottom, top)
        valid = True
        if not protect:
            valid = self.validate()
        return top, valid

    def FKSciRaphson(self, L, bottom_plate_pos = None, reverse = False, protect = False):
        """
        Use Python's Scipy module to calculate forward kinematics. Takes in length list,
        optionally bottom position, reverse parameter, and protection
        Args:
            L (ndarray(Float)): Description of parameter `L`.
            bottom_plate_pos (tm): Description of parameter `bottom_plate_pos`.
            reverse (Bool): Description of parameter `reverse`.
            protect (Bool): Description of parameter `protect`.

        Returns:
            type: Description of returned object.

        """
        L = L.reshape((6, 1))
        mag = lambda x : abs(x[0]) + abs(x[1])+ abs(x[2]) + abs(x[3]) + abs(x[4]) + abs(x[5])
        fk = lambda x : mag(self.IKHelper(bottom_plate_pos, tm(x), protect = True)[0] - L).flatten()
        jac = lambda x : (self.inverseJacobianSpace(bottom_plate_pos, tm(x)))
        x0 = (self.getBottomT() @ self.nominal_plate_transform).TAA.flatten()

        root = sci.optimize.minimize(fk, x0).x
        #disp(root, "ROOT")
        self.IK(bottom_plate_pos, tm(root), protect = True)
        return bottom_plate_pos, tm(root)

    def simplifiedRaphson(self, L, bottom_plate_pos = None, reverse = False, protect = False):
        """
        Follow the method in the Parallel Robotics Textbook

        Args:
            L (ndarray(Float)): Description of parameter `L`.
            bottom_plate_pos (tm): Description of parameter `bottom_plate_pos`.
            reverse (Bool): Description of parameter `reverse`.
            protect (Bool): Description of parameter `protect`.

        Returns:
            type: Description of returned object.

        """
        tol_f = 1e-4;
        tol_a = 1e-4;
        #iteration limits
        max_iterations = 1e4

        if bottom_plate_pos == None:
            bottom_plate_pos = self.bottom_plate_pos

        x = self.getTopT().copy()
        iter = 0
        success = False
        while not success and iter < max_iterations:
            x = x + self.inverseJacobianSpace(bottom_plate_pos, x ) @ (L -
                self.IK(top_plate_pos = x, protect = protect))
            x.AngleMod()
            #disp(x)
            if np.all(abs(x[0:3]) < tol_f) and np.all(abs(x[3:6]) < tol_a):
                success = True
            iter+=1

        if iter == max_iterations:
            print("Failed to Converge")

        return tm(x)



    def FKSolve(self, L, bottom_plate_pos = None, reverse = False, protect = False):
        """
        Older version of python solver, no jacobian used. Takes in length list,
        optionally bottom position, reverse parameter, and protection

        Args:
            L (ndarray(Float)): Description of parameter `L`.
            bottom_plate_pos (tm): Description of parameter `bottom_plate_pos`.
            reverse (Bool): Description of parameter `reverse`.
            protect (Bool): Description of parameter `protect`.

        Returns:
            type: Description of returned object.

        """
        #Do SPFK with scipy inbuilt solvers. Way less speedy o
        #Or accurate than Raphson, but much simpler to look at
        L = L.reshape((6, 1))
        self.lengths = L.reshape((6, 1)).copy()
        #jac = lambda x : self.inverseJacobianSpace(top_plate_pos = x)

        #Slightly different if the platform is supposed to be "reversed"
        if reverse:
            if bottom_plate_pos == None:
                top_plate_pos = self.getTopT()
            else:
                top_plate_pos = bottom_plate_pos
            fk = lambda x : (self.IK(tm(x), top_plate_pos, protect = True) - L).reshape((6))
            sol = tm(sci.optimize.fsolve(fk, self.getTopT().gTAA()))
            #self.top_plate_pos = bottom_plate_pos
        else:
            #General calls will go here.
            if bottom_plate_pos == None:
                #If no bottom pose is supplied, use the last known.
                bottom_plate_pos = self.getBottomT()
            #Find top pose that produces the desired leg lengths.
            fk = lambda x : (self.IKHelper(bottom_plate_pos, tm(x),
                protect = True)[0] - L).reshape((6))
            sol = tm(sci.optimize.fsolve(fk, self.getTopT().TAA))
            #self.bottom_plate_pos = bottom_plate_pos

        #If not "Protected" from recursion, call IK.
        if not protect:
            self.IK(protect = True)
        return bottom_plate_pos, sol


    def FKRaphson(self, L, bottom_plate_pos =None, reverse = False, protect = False):
        """
        FK Solver
        Adapted from the work done by
        #http://jak-o-shadows.github.io/electronics/stewart-gough/stewart-gough.html
        Args:
            L (ndarray(Float)): Description of parameter `L`.
            bottom_plate_pos (tm): Description of parameter `bottom_plate_pos`.
            reverse (Bool): Description of parameter `reverse`.
            protect (Bool): Description of parameter `protect`.

        Returns:
            type: Description of returned object.

        """
        if self.debug:
            disp("Starting Raphson FK")
        #^Look here for the original code and paper describing how this works.
        if bottom_plate_pos == None:
            bottom_plate_pos = self.getBottomT()
        success = True
        L = L.reshape((6))
        self.lengths = L.reshape((6, 1)).copy()

        bottom_plate_pos_backup = bottom_plate_pos.copy()
            # @ tm([0, 0, self.bottom_plate_thickness, 0, 0, 0])
        bottom_plate_pos = np.eye(4)
        #bottom_plate_pos = bottom_plate_pos_backup.copy()
        #newton-raphson tolerances
        #iteration limits
        iteration = 0

        #Initial Guess Position
        #a = fsr.TMtoTAA(bottom_plate_pos @
        #   fsr.TM([0, 0, self.nominal_height, 0, 0, 0])).reshape((6))
        #disp(a, "Attempt")
        try:
            #ap = (fsr.LocalToGlobal(tm([0, 0, self.nominal_height, 0, 0, 0]), tm()))
            ap = (fsr.LocalToGlobal(self.current_plate_transform_local, tm())).gTAA().reshape((6))
            a = np.zeros((6))
            for i in range(6):
                a[i] = ap[i]

            #Call the actual algorithm from the high performance faser library
            #Pass in initial lengths, guess, bottom and top plate positions,
            #max iterations, tolerances, and minimum leg lengths
            a, iteration = fmr.SPFKinSpaceR(bottom_plate_pos, L, a,
                self.bottom_joints_init, self.top_joints_init,
                self.max_iterations, self.tol_f, self.tol_a, self.leg_ext_min)

            #If the algorithm failed, try again, but this time set initial position to neutral
            if iteration == self.max_iterations:

                a = np.zeros((6))
                a[2] = self.nominal_height
                a, iteration = fmr.SPFKinSpaceR(bottom_plate_pos, L, a,
                    self.bottom_joints_init, self.top_joints_init,
                    self.max_iterations, self.tol_f, self.tol_a, self.leg_ext_min)
                if iteration == self.max_iterations:
                    if self.debug:
                        print("Raphson Failed to Converge")
                    self.fail_count += .1
                    self.IK(bottom_plate_pos_backup,
                        bottom_plate_pos_backup @ self.nominal_plate_transform, protect = True)
                    return self.getBottomT(), self.getTopT()

            #Otherwise return the calculated end effector position
            #coords =tm(bottom_plate_pos_backup @ fsr.TAAtoTM(a.reshape((6, 1))))
            coords = bottom_plate_pos_backup @ tm(a)
            # @ tm([0, 0, self.top_plate_thickness, 0, 0, 0])

            #Disabling these cause unknown issues so far.
            #self.bottom_plate_pos = bottom_plate_pos_backup
            #self.top_plate_pos = coords


            self.IKHelper(bottom_plate_pos_backup, coords, protect = True)
            self.bottom_plate_pos = bottom_plate_pos_backup
            #@ tm([0, 0, self.bottom_plate_thickness, 0, 0, 0])
            self.top_plate_pos = coords #@ tm([0, 0, self.top_plate_thickness, 0, 0, 0])
            if self.debug:
                disp("Returning from Raphson FK")
            return bottom_plate_pos_backup, tm(coords)
        except Exception as e:

            if self.debug:
                disp("Raphson FK Failed due to: " + str(e))
            self.fail_count+=1
            return self.FKSciRaphson(L, bottom_plate_pos_backup, reverse, protect)


    def lambdaTopPlateReorientation(self, stopt):
        """
        Only used as an assistance function for fixing plate alignment

        Args:
            stopt (tm): Description of parameter `stopt`.

        Returns:
            type: Description of returned object.

        """
        reorient_helper_1 = fsr.LocalToGlobal(stopt, self.reorients[0])
        reorient_helper_2 = fsr.LocalToGlobal(stopt, self.reorients[1])
        reorient_helper_3 = fsr.LocalToGlobal(stopt, self.reorients[2])

        d1 = fsr.Distance(reorient_helper_1,
            tm([self.top_joints_space[0, 0],
            self.top_joints_space[1, 0],
            self.top_joints_space[2, 0], 0, 0, 0]))
        d2 = fsr.Distance(reorient_helper_2,
            tm([self.top_joints_space[0, 2],
            self.top_joints_space[1, 2],
            self.top_joints_space[2, 2], 0, 0, 0]))
        d3 = fsr.Distance(reorient_helper_3,
            tm([self.top_joints_space[0, 4],
            self.top_joints_space[1, 4],
            self.top_joints_space[2, 4], 0, 0, 0]))
        return np.array([d1 , d2 , d3])

    def reorientTopPlate(self):
        """
        Subfunction of fixUpsideDown,
        responsible for orienting the top plate transform after mirroring
        Returns:
            type: Description of returned object.

        """
        top_true = self.getTopT() @ tm([0, 0, -self.top_plate_thickness, 0, 0, 0])
        res = lambda x : self.lambdaTopPlateReorientation(
            tm([top_true[0], top_true[1], top_true[2], x[0], x[1], x[2]]))
        x_init = self.getTopT()[3:6].flatten()
        solution = sci.optimize.fsolve(res, x_init)
        top_true[3:6] = solution
        self.top_plate_pos = top_true @ tm([0, 0, self.top_plate_thickness, 0, 0, 0])
        #disp(self.lambdaTopPlateReorientation(self.getTopT() @
        #   tm([0, 0, -self.top_plate_thickness, 0, 0, 0])))


    def fixUpsideDown(self):
        """
        In situations where the top plate is inverted underneath
        the bottom plate, yet lengths are valid,
        This function can be used to mirror all the joint locations and "fix" the resultant problem

        Returns:
            type: Description of returned object.
        """
        for num in range(6):
            #reversable = fsr.GlobalToLocal(tm([self.top_joints_space[0, num],
            #    self.top_joints_space[1, num], self.top_joints_space[2, num], 0, 0, 0]),
            #    tm([self.bottom_joints_space[0, num],
            #    self.bottom_joints_space[1, num],
            #    self.bottom_joints_space[2, num], 0, 0, 0]))
            #newTJ = tm([self.bottom_joints_space[0, num],
            #    self.bottom_joints_space[1, num],
            #    self.bottom_joints_space[2, num], 0, 0, 0]) @ reversable
            newTJ = fsr.Mirror(self.getBottomT() @ tm([0, 0, -self.bottom_plate_thickness, 0, 0, 0]),
                tm([self.top_joints_space[0, num],
                self.top_joints_space[1, num],
                self.top_joints_space[2, num], 0, 0, 0]))
            self.top_joints_space[0, num] = newTJ[0]
            self.top_joints_space[1, num] = newTJ[1]
            self.top_joints_space[2, num] = newTJ[2]
            self.lengths[num] = fsr.Distance(
                self.top_joints_space[:, num], self.bottom_joints_space[:, num])
        top_true = fsr.Mirror(self.getBottomT() @ tm([0, 0, -self.bottom_plate_thickness, 0, 0, 0]),
            self.getTopT() @ tm([0, 0, -self.top_plate_thickness, 0, 0, 0]))
        top_true[3:6] = self.getTopT()[3:6] * -1
        self.top_plate_pos = top_true @ tm([0, 0, self.top_plate_thickness, 0, 0, 0])
        self.reorientTopPlate()

    def validateLegs(self, valid = True, donothing = False):
        """

        Args:
            valid (Bool): Description of parameter `valid`.
            donothing (Bool): Description of parameter `donothing`.

        Returns:
            type: Description of returned object.

        """
        if self.validation_settings[0]:
            temp_valid = self.legLengthConstraint(donothing)
            valid = valid and temp_valid
            if not temp_valid:
                self.validation_error += "Leg Length Constraint Violated "
            if not temp_valid and not donothing:
                if self.debug:
                    disp("Executing Length Corrective Action...")
                self.lengthCorrectiveAction()
                valid = self.validate(True, 1)
        return valid

    def validateContinuousTranslation(self, valid=True, donothing = False):
        """

        Args:
            valid (Bool): Description of parameter `valid`.
            donothing (Bool): Description of parameter `donothing`.

        Returns:
            type: Description of returned object.

        """
        if self.validation_settings[1]:
            temp_valid = self.continuousTranslationConstraint()
            valid = valid and temp_valid
            if not temp_valid:
                self.validation_error += "Platform Inversion Constraint Violated "
            if not temp_valid and not donothing:
                if self.debug:
                    disp("Executing Continuous Translation Corrective Action...")
                self.continuousTranslationCorrectiveAction()
                valid = self.validate(True, 2)
        return valid
    def validateInteriorAngles(self, valid = True, donothing = False):
        """

        Args:
            valid (Bool): Description of parameter `valid`.
            donothing (Bool): Description of parameter `donothing`.

        Returns:
            type: Description of returned object.

        """
        if self.validation_settings[2]:
            temp_valid = self.interiorAnglesConstraint()
            valid = valid and temp_valid
            if not temp_valid:
                self.validation_error += "Interior Angles Constraint Violated "
            if not temp_valid and not donothing:
                if self.debug:
                    disp("Executing Interior Angles Corrective Action...")
                self.IK(self.getBottomT(), self.getBottomT() @
                    self.nominal_plate_transform, protect = True)
                valid = self.validate(True, 3)
        return valid

    def validatePlateRotation(self, valid = True, donothing = False):
        """

        Args:
            valid (Bool): Description of parameter `valid`.
            donothing (Bool): Description of parameter `donothing`.

        Returns:
            type: Description of returned object.

        """
        if self.validation_settings[3]:
            temp_valid = self.plateRotationConstraint()
            valid = valid and temp_valid
            if not temp_valid:
                self.validation_error += "Plate Tilt/Rotate Constraint Violated "
            if not temp_valid and not donothing:
                if self.debug:
                    disp("Executing Plate Rotation Corrective Action By Resetting Platform")
                #disp(self.nominal_plate_transform)
                self.IK(self.getBottomT(),(self.getBottomT() @
                    self.nominal_plate_transform), protect = True)
                valid = self.validate(True, 4)
        return valid

    def validate(self, donothing = False, validation_limit = 4):
        """
        Validate the current configuration of the stewart platform

        Args:
            donothing (Bool): Description of parameter `donothing`.
            validation_limit (Int): Description of parameter `validation_limit`.

        Returns:
            type: Description of returned object.

        """
        valid = True #innocent until proven INVALID
        #if self.debug:
        #    disp("Validating")
        #First check to make sure leg lengths are not exceeding limit points
        if fsr.Distance(self.getTopT(), self.getBottomT()) > 2 * self.nominal_height:
            valid = False

        if validation_limit > 0: valid = self.validateLegs(valid, donothing)
        if validation_limit > 1: valid = self.validateContinuousTranslation(valid, donothing)
        if validation_limit > 2: valid = self.validateInteriorAngles(valid, donothing)
        if validation_limit > 3: valid = self.validatePlateRotation(valid, donothing)

        if valid:
            self.validation_error = ""

        return valid

    def plateRotationConstraint(self):
        """

        Returns:
            type: Description of returned object.

        """
        valid = True
        for i in range(3):
            if self.current_plate_transform_local.gTM()[i, i] <= self.plate_rotation_limit - .0001:
                if self.debug:
                    disp(self.current_plate_transform_local.gTM(), "Erroneous TM")
                    print([self.current_plate_transform_local.gTM()[i, i],
                        self.plate_rotation_limit])
                valid = False
        return valid

    def legLengthConstraint(self, donothing):
        """
        Evaluate Leg Length Limitations of Stewart Platform

        Args:
            donothing (Bool): Description of parameter `donothing`.

        Returns:
            type: Description of returned object.

        """
        valid = True
        if(np.any(self.lengths < self.leg_ext_min) or np.any(self.lengths > self.leg_ext_max)):
            valid = False
        return valid

    def rescaleLegLengths(self, current_leg_min, current_leg_max):
        """

        Args:
            current_leg_min (Float): Description of parameter `current_leg_min`.
            current_leg_max (Float): Description of parameter `current_leg_max`.

        Returns:
            type: Description of returned object.

        """

        for i in range(6):
            self.lengths[i] = ((self.lengths[i]-current_leg_min)/
                (current_leg_max-current_leg_min) *
                (min(self.leg_ext_max, current_leg_max) -
                max(self.leg_ext_min, current_leg_min)) +
                max(self.leg_ext_min, current_leg_min))

    def addLegsToMinimum(self, current_leg_min, current_leg_max):
        """

        Args:
            current_leg_min (Float): Description of parameter `current_leg_min`.
            current_leg_max (Float): Description of parameter `current_leg_max`.

        Returns:
            type: Description of returned object.

        """
        boostamt = ((self.leg_ext_min-current_leg_min)+self.leg_ext_safety)
        if self.debug:
            print("Boost Amount: " + str(boostamt))
        self.lengths += boostamt

    def subLegsToMaximum(self, current_leg_min, current_leg_max):
        """

        Args:
            current_leg_min (Float): Description of parameter `current_leg_min`.
            current_leg_max (Float): Description of parameter `current_leg_max`.

        Returns:
            type: Description of returned object.

        """
        #print([current_leg_max, self.leg_ext_max, current_leg_min,
        #    self.leg_ext_min, current_leg_max -
        #    (current_leg_max - self.leg_ext_max + self.leg_ext_safety)])
        self.lengths -= ((current_leg_max - self.leg_ext_max)+self.leg_ext_safety)
        #print(self.lengths)
    def lengthCorrectiveAction(self):
        """
        Make an attempt to correct leg lengths that are out of bounds.
        Will frequently result in a home-like position
        Returns:
            type: Description of returned object.

        """
        if self.debug:
            disp(self.lengths, "Lengths Pre Correction")
            disp(self.lengths[np.where(self.lengths > self.leg_ext_max)], "over max")
            disp(self.lengths[np.where(self.lengths < self.leg_ext_min)], "below min")

        current_leg_min = min(self.lengths.flatten())
        current_leg_max = max(self.lengths.flatten())

        #for i in range(6):
        #    self.lengths[i] = ((self.lengths[i]-current_leg_min)/
        #    (current_leg_max-current_leg_min) *
        #    (min(self.leg_ext_max, current_leg_max) -
        #    max(self.leg_ext_min, current_leg_min)) +
        #    max(self.leg_ext_min, current_leg_min))
        if current_leg_min < self.leg_ext_min and current_leg_max > self.leg_ext_max:
            self.rescaleLegLengths(current_leg_min, current_leg_max)
            self.validation_error+= " CMethod: Rescale, "
        elif (current_leg_min < self.leg_ext_min and
            current_leg_max + (self.leg_ext_min - current_leg_min) +
            self.leg_ext_safety < self.leg_ext_max):
            self.addLegsToMinimum(current_leg_min, current_leg_max)
            self.validation_error+= " CMethod: Boost, "
        elif (current_leg_max > self.leg_ext_max and
            current_leg_min - (current_leg_max - self.leg_ext_max) -
            self.leg_ext_safety > self.leg_ext_min):
            self.validation_error+= " CMethod: Subract, "
            self.subLegsToMaximum(current_leg_min, current_leg_max)
        else:
            self.rescaleLegLengths(current_leg_min, current_leg_max)
            self.validation_error+= " CMethod: Unknown Rescale, "

        #self.lengths[np.where(self.lengths > self.leg_ext_max)] = self.leg_ext_max
        #self.lengths[np.where(self.lengths < self.leg_ext_min)] = self.leg_ext_min
        if self.debug:
            disp(self.lengths, "Corrected Lengths")
        #disp("HEre's what happened")
        self.FK(self.lengths.copy(), protect = True)
        #print(self.lengths)

    def continuousTranslationConstraint(self):
        """
        Ensure that the plate is above the prior

        Returns:
            type: Description of returned object.

        """
        valid = True
        bot = self.getBottomT()
        for i in range(6):
            if fsr.GlobalToLocal(self.getBottomT(), self.getTopT())[2] < 0:
                valid = False
        return valid

    def continuousTranslationCorrectiveAction(self):
        """

        Returns:
            type: Description of returned object.

        """
        self.IK(top_plate_pos = self.getBottomT() @ self.nominal_plate_transform, protect = True)

    def interiorAnglesConstraint(self):
        """
        Ensures no invalid internal angles

        Returns:
            type: Description of returned object.

        """
        angles = abs(self.getJointAnglesFromNorm())
        if(np.any(np.isnan(angles))):
            return False
        if(np.any(angles > self.joint_deflection_max)):
            return False
        return True

    def getJointAnglesFromNorm(self):
        """
        Returns the angular deviation of each angle socket from its nominal position in radians

        Returns:
            type: Description of returned object.

        """
        delta_angles_top = np.zeros((6))
        delta_angles_bottom = np.zeros((6))
        bottom_plate_transform = self.getBottomT()
        top_plate_transform = self.getTopT()
        for i in range(6):

                top_joint_i = tm([
                    self.top_joints_space.T[i][0],
                    self.top_joints_space.T[i][1],
                    self.top_joints_space.T[i][2],
                    top_plate_transform[3],
                    top_plate_transform[4],
                    top_plate_transform[5]])
                bottom_joint_i = tm([
                    self.bottom_joints_space.T[i][0],
                    self.bottom_joints_space.T[i][1],
                    self.bottom_joints_space.T[i][2],
                    bottom_plate_transform[3],
                    bottom_plate_transform[4],
                    bottom_plate_transform[5]])

                #We have the relative positions to the top plate
                #   of the bottom joints (bottom angles) in home pose
                #We have the relative positions to the bottom plate of
                #   the top joints (bottom_joint_angles_init) in home pose
                bottom_to_top_local_home = self.bottom_joint_angles_init[i].copy()
                top_to_bottom_local_home = self.bottom_joint_angles[i].copy()

                #We acquire the current relative (local positions of each)
                bottom_to_top_local = fsr.GlobalToLocal(self.getBottomT(), top_joint_i)
                top_to_bottom_local = fsr.GlobalToLocal(self.getTopT(), bottom_joint_i)

                #We acquire the base positions of each joint
                bottom_to_bottom_local = fsr.GlobalToLocal(self.getBottomT(), bottom_joint_i)
                top_to_top_local = fsr.GlobalToLocal(self.getTopT(), top_joint_i)

                delta_angles_bottom[i] = fsr.AngleBetween(
                    bottom_to_top_local,
                    bottom_to_bottom_local,
                    bottom_to_top_local_home)
                delta_angles_top[i] = fsr.AngleBetween(
                    top_to_bottom_local,
                    top_to_top_local,
                    top_to_bottom_local_home)

            #DeltAnglesA are the Angles From Norm Bottom
            #DeltAnglesB are the Angles from Norm TOp
        return np.hstack((delta_angles_bottom, delta_angles_top))

    def getJointAnglesFromVertical(self):
        """

        Returns:
            type: Description of returned object.

        """
        top_down = np.zeros((6))
        bottom_up = np.zeros((6))
        for i in range(6):
            top_joints_temp = self.top_joints_space[:, i].copy().flatten()
            top_joints_temp[2] = 0
            bottom_joints_temp = self.bottom_joints_space[:, i].copy().flatten()
            bottom_joints_temp[2] = bottom_joints_temp[2] + 1
            angle = fsr.AngleBetween(
                self.bottom_joints_space[:, i],
                self.top_joints_space[:, i],
                top_joints_temp)
            angle_up = fsr.AngleBetween(
                self.top_joints_space[:, i],
                self.bottom_joints_space[:, i],
                bottom_joints_temp)
            top_down[i] = angle
            bottom_up[i] = angle_up
        return top_down, bottom_up

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
        """

        Args:
            tau (ndarray(Float)): Description of parameter `tau`.

        Returns:
            type: Description of returned object.

        """
        vertical_components = np.zeros((6))
        horizontal_components = np.zeros((6))
        for i in range(6):
            top_joint = self.top_joints_space[:, i].copy().flatten()
            top_joint[2] = 0
            angle = fsr.AngleBetween(
                self.bottom_joints_space[:, i],
                self.top_joints_space[:, i],
                top_joint)
            vertical_force = tau[i] * np.sin(angle)
            horizontal_force = tau[i] * np.cos(angle)
            vertical_components[i] = vertical_force
            horizontal_components[i] = horizontal_force
        return vertical_components, horizontal_components

    def bottomTopCheck(self, bottom_plate_pos, top_plate_pos):
        """
        Checks to make sure that a bottom and top provided are not null

        Args:
            bottom_plate_pos (tm): Description of parameter `bottom_plate_pos`.
            top_plate_pos (tm): Description of parameter `top_plate_pos`.

        Returns:
            type: Description of returned object.

        """
        if bottom_plate_pos == None:
            bottom_plate_pos = self.getBottomT()
        if top_plate_pos == None:
            top_plate_pos = self.getTopT()
        return bottom_plate_pos, top_plate_pos

    def jacobianSpace(self, bottom_plate_pos = None, top_plate_pos = None):
        """
        Calculates space jacobian for stewart platform. Takes in bottom transform and top transform

        Args:
            bottom_plate_pos (tm): Description of parameter `bottom_plate_pos`.
            top_plate_pos (tm): Description of parameter `top_plate_pos`.

        Returns:
            type: Description of returned object.

        """
        #If not supplied paramters, draw from stored values
        bottom_plate_pos, top_plate_pos = self.bottomTopCheck(bottom_plate_pos, top_plate_pos)
        #Just invert the inverted
        inverse_jacobian = self.inverseJacobianSpace(bottom_plate_pos, top_plate_pos)
        return ling.pinv(inverse_jacobian)


    def inverseJacobianSpace(self, bottom_plate_pos = None, top_plate_pos = None, protect = True):
        """
        Calculates Inverse Jacobian for stewart platform. Takes in bottom and top transforms

        Args:
            bottom_plate_pos (tm): Description of parameter `bottom_plate_pos`.
            top_plate_pos (tm): Description of parameter `top_plate_pos`.
            protect (Bool): Description of parameter `protect`.

        Returns:
            type: Description of returned object.

        """
        #Ensure everything is kosher with the plates
        bottom_plate_pos, top_plate_pos = self.bottomTopCheck(bottom_plate_pos, top_plate_pos)

        #Store old values
        old_bottom_plate_transform = self.getBottomT()
        old_top_plate_transform = self.getTopT()

        #Perform IK on bottom and top
        self.IK(bottom_plate_pos, top_plate_pos, protect = protect)

        #Create Jacobian
        inverse_jacobian_transpose = np.zeros((6, 6))
        for i in range(6):
            #todo check sign on nim,
            ni = fmr.Normalize(self.top_joints_space[:, i]-self.bottom_joints_space[:, i])
             #Reverse for upward forces?
            qi = self.bottom_joints_space[:, i]
            col = np.hstack((np.cross(qi, ni), ni))
            inverse_jacobian_transpose[:, i] = col
        inverse_jacobian = inverse_jacobian_transpose.T

        #Restore original Values
        self.IK(old_bottom_plate_transform, old_top_plate_transform, protect = protect)
        return inverse_jacobian

    #Returns Top Down Jacobian instead of Bottom Up
    def altInverseJacobianSpace(self,
        bottom_plate_pos = None, top_plate_pos = None, protect = True):
        """
        Returns top down jacobian instead of bottom up

        Args:
            bottom_plate_pos (tm): Description of parameter `bottom_plate_pos`.
            top_plate_pos (tm): Description of parameter `top_plate_pos`.
            protect (Bool): Description of parameter `protect`.

        Returns:
            type: Description of returned object.

        """
        bottom_plate_pos, top_plate_pos = self.bottomTopCheck(bottom_plate_pos, top_plate_pos)
        old_bottom_plate_transform = copy.copy(bottom_plate_pos)
        old_top_plate_transform = copy.copy(top_plate_pos)
        self.IK(bottom_plate_pos, top_plate_pos)
        inverse_jacobian_transpose = np.zeros((6, 6))
        for i in range(6):
            ni = fmr.Normalize(self.bottom_joints_space[:, i]-self.top_joints_space[:, i])
            qi = self.top_joints_space[:, i]
            inverse_jacobian_transpose[:, i] = np.hstack((np.cross(qi, ni), ni))
        inverse_jacobian = inverse_jacobian_transpose.conj().transpose()

        self.IKHelper(old_bottom_plate_transform, old_top_plate_transform)

        return inverse_jacobian

    #Adds in actuator and plate forces, useful for finding forces on a full stack assembler
    def carryMassCalc(self, twrench, protect = False):
        """
        Calculates the forces on each leg given their masses,
        masses of plates, and a wrench on the end effector.
        Args:
            twrench (ndarray(Float)): Description of parameter `twrench`.
            protect (Bool): Description of parameter `protect`.

        Returns:
            type: Description of returned object.

        """
        wrench = twrench.copy()
        wrench = wrench + fsr.GenForceWrench(self.getTopT(),
            self.top_plate_newton_force, self.dir)
        tau = self.measureForcesFromWrenchEE(self.getBottomT(),
            self.getTopT(), wrench, protect = protect)
        for i in range(6):
            #print(self.getActuatorLoc(i, 't'))
            wrench += fsr.GenForceWrench(self.getActuatorLoc(i, 't'),
                self.act_shaft_newton_force, self.dir)
            wrench += fsr.GenForceWrench(self.getActuatorLoc(i, 'b'),
                self.act_motor_newton_force, self.dir)
        wrench = wrench + fsr.GenForceWrench(self.getBottomT(),
            self.bottom_plate_newton_force, self.dir)
        return tau, wrench

    def carryMassCalcLocal(self, twrench, protect = False):
        """

        Args:
            twrench (ndarray(Float)): Description of parameter `twrench`.
            protect (Bool): Description of parameter `protect`.

        Returns:
            type: Description of returned object.

        """
        #We will here assume that the wrench is in the local frame of the top platform.
        wrench = twrench.copy()
        wrench = wrench + fsr.GenForceWrench(tm(), self.top_plate_newton_force, self.dir)
        tau = self.measureForcesAtEENew(wrench, protect = protect)
        wrench_local_frame = fsr.TransformWrenchFrame(wrench, self.getTopT(), self.getBottomT())

        for i in range(6):
            #print(self.getActuatorLoc(i, 't'))
            #The following representations are equivalent.
            wrench_local_frame += fsr.GenForceWrench(fsr.GlobalToLocal(self.getActuatorLoc(i, 't'),
                self.getBottomT()), self.act_shaft_newton_force, self.dir)
            wrench_local_frame += fsr.GenForceWrench(fsr.GlobalToLocal(self.getActuatorLoc(i, 'b'),
                self.getBottomT()), self.act_motor_newton_force, self.dir)
            #wrench_local_frame += fsr.TransformWrenchFrame(fsr.GenForceWrench(tm(),
            #    self.act_shaft_newton_force, self.dir),
            #   self.getActuatorLoc(i, 't'), self.getBottomT())
            #wrench_local_frame += fsr.TransformWrenchFrame(fsr.GenForceWrench(tm(),
            #    self.act_motor_newton_force, self.dir),
            #   self.getActuatorLoc(i, 'b'), self.getBottomT())
        wrench_local_frame = wrench_local_frame + fsr.GenForceWrench(tm(),
            self.bottom_plate_newton_force, self.dir)
        return tau, wrench_local_frame

    def measureForcesAtEENew(self, wrench, protect = False):
        """

        Args:
            wrench (ndarray(Float)): Description of parameter `wrench`.
            protect (Bool): Description of parameter `protect`.

        Returns:
            type: Description of returned object.

        """
        jacobian_space = ling.pinv(
            self.inverseJacobianSpace(self.getBottomT(), self.getTopT(), protect = protect))
        tau = jacobian_space.T @ wrench
        self.leg_forces = tau
        return tau

    def carryMassCalcUp(self, twrench, protect = False):
        """

        Args:
            twrench (ndarray(Float)): Description of parameter `twrench`.
            protect (Bool): Description of parameter `protect`.

        Returns:
            type: Description of returned object.

        """
        wrench = twrench.copy()
        wrench = wrench + fsr.GenForceWrench(self.getBottomT(),
            self.bottom_plate_mass * self.grav, np.array([0, 0, -1]))
        tau = self.measureForcesFromBottomEE(
            self.getBottomT(), self.getTopT(), wrench, protect = protect)
        for i in range(6):
            wrench += fsr.GenForceWrench(
                self.getActuatorLoc(i, 't'), self.act_shaft_mass * self.grav, np.array([0, 0, -1]))
            wrench += fsr.GenForceWrench(
                self.getActuatorLoc(i, 'b'), self.act_motor_mass * self.grav, np.array([0, 0, -1]))
        wrench = wrench + fsr.GenForceWrench(
            self.getTopT(), self.top_plate_mass * self.grav, np.array([0, 0, -1]))
        return tau, wrench

    #Get Force wrench from the End Effector Force
    def measureForcesFromWrenchEE(self, bottom_plate_pos = np.zeros((1)),
        top_plate_pos = np.zeros((1)), top_plate_wrench = np.zeros((1)), protect = True):
        """
        Calculates forces on legs given end effector wrench

        Args:
            bottom_plate_pos (tm): Description of parameter `bottom_plate_pos`.
            top_plate_pos (tm): Description of parameter `top_plate_pos`.
            top_plate_wrench (ndarray(Float)): Description of parameter `top_plate_wrench`.
            protect (Bool): Description of parameter `protect`.

        Returns:
            measureForcesFromWrenchEE(self, bottom_plate_pos =: Description of returned object.

        """
        bottom_plate_pos, top_plate_pos = self.bottomTopCheck(bottom_plate_pos, top_plate_pos)
        if top_plate_wrench.size < 6:
            disp("Please Enter a Wrench")
        #top_wrench = fmr.Adjoint(ling.inv(top_plate_pos)).conj().transpose() @ top_plate_wrench
        #Modern Robotics 3.95 Fb = Ad(Tba)^T * Fa
        #top_wrench = top_plate_pos.inv().Adjoint().T @ top_plate_wrench
        top_wrench = fsr.TransformWrenchFrame(top_plate_wrench, tm(), top_plate_pos)
        jacobian_space = ling.pinv(
            self.inverseJacobianSpace(bottom_plate_pos, top_plate_pos, protect = protect))
        tau = jacobian_space.T @ top_wrench
        self.leg_forces = tau
        return tau

    def measureForcesFromBottomEE(self, bottom_plate_pos = np.zeros((1)),
        top_plate_pos = np.zeros((1)), top_plate_wrench = np.zeros((1)), protect = True):
        """
        Calculates forces on legs given end effector wrench

        Args:
            bottom_plate_pos (tm): Description of parameter `bottom_plate_pos`.
            top_plate_pos (tm): Description of parameter `top_plate_pos`.
            top_plate_wrench (ndarray(Float)): Description of parameter `top_plate_wrench`.
            protect (Bool): Description of parameter `protect`.

        Returns:
            measureForcesFromBottomEE(self, bottom_plate_pos =: Description of returned object.

        """
        bottom_plate_pos, top_plate_pos = self._bttomTopCheck(bottom_plate_pos, top_plate_pos)
        if top_plate_wrench.size < 6:
            disp("Please Enter a Wrench")
        #top_wrench = fmr.Adjoint(ling.inv(top_plate_pos)).conj().transpose() @ top_plate_wrench
        bottom_wrench = bottom_plate_pos.inv().Adjoint().T @ top_plate_wrench
        jacobian_space = ling.pinv(
            self.inverseJacobianSpace(bottom_plate_pos, top_plate_pos, protect = protect))
        tau = jacobian_space.T @ bottom_wrench
        self.leg_forces = tau
        return tau

    def wrenchEEFromMeasuredForces(self, bottom_plate_pos, top_plate_pos, tau):
        """
        Calculates wrench on end effector from leg forces

        Args:
            bottom_plate_pos (tm): Description of parameter `bottom_plate_pos`.
            top_plate_pos (tm): Description of parameter `top_plate_pos`.
            tau (ndarray(Float)): Description of parameter `tau`.

        Returns:
            type: Description of returned object.

        """
        self.leg_forces = tau
        jacobian_space = ling.pinv(self.inverseJacobianSpace(bottom_plate_pos, top_plate_pos))
        top_wrench = ling.inv(jacobian_space.conj().transpose()) @ tau
        #self.top_plate_wrench = fmr.Adjoint(top_plate_pos).conj().transpose() @ top_wrench
        self.top_plate_wrench = top_plate_pos.Adjoint().conj().transpose() @ top_wrench
        return self.top_plate_wrench, top_wrench, jacobian_space

    def wrenchBottomFromMeasuredForces(self, bottom_plate_pos, top_plate_pos, tau):
        """
        Unused. Calculates wrench on the bottom plate from leg forces

        Args:
            bottom_plate_pos (tm): Description of parameter `bottom_plate_pos`.
            top_plate_pos (tm): Description of parameter `top_plate_pos`.
            tau (ndarray(Float)): Description of parameter `tau`.

        Returns:
            type: Description of returned object.

        """
        self.leg_forces = tau
        jacobian_space = ling.pinv(self.altInverseJacobianSpace(bottom_plate_pos, top_plate_pos))
        bottom_wrench = ling.inv(jacobian_space.conj().transpose()) @ tau
        #self.bottom_plate_wrench = fmr.Adjoint(bottom_plate_pos).conj().transpose() @ bottom_wrench
        self.bottom_plate_wrench = bottom_plate_pos.Adjoint().conj().transpose() @ bottom_wrench
        return self.bottom_plate_wrench, bottom_wrench, jacobian_space

    def sumActuatorWrenches(self, forces = None):
        """

        Args:
            forces (ndarray(Float)): Description of parameter `forces`.

        Returns:
            type: Description of returned object.

        """
        if forces is None:
            forces = self.leg_forces

        wrench = fsr.GenForceWrench(tm(), 0, [0, 0, -1])
        for i in range(6):
            unit_vector = fmr.Normalize(self.bottom_joints_space[:, i]-self.top_joints_space[:, i])
            wrench += fsr.GenForceWrench(self.top_joints_space[:, i], float(forces[i]), unit_vector)
        #wrench = fsr.TransformWrenchFrame(wrench, tm(), self.getTopT())
        return wrench


    def move(self, T, protect = False):
        """
        Move entire Assembler Stack to another location and orientation
        This function and syntax are shared between all kinematic structures.
        Args:
            T (tm): Description of parameter `T`.
            protect (Bool): Description of parameter `protect`.

        Returns:
            type: Description of returned object.

        """
        #Moves the base of the stewart platform to a new location


        self.current_plate_transform_local = fsr.GlobalToLocal(self.getBottomT(), self.getTopT())
        self.bottom_plate_pos = T.copy()
        self.IK(
            top_plate_pos = fsr.LocalToGlobal(self.getBottomT(), self.current_plate_transform_local),
            protect = protect)

    def printOutOfDateFunction(self, old_name, use_name):
        """

        Args:
            old_name (String): Description of parameter `old_name`.
            use_name (String): Description of parameter `use_name`.

        Returns:
            type: Description of returned object.

        """
        print(old_name + " is deprecated. Please use " + use_name + " instead.")

    def SetMasses(self, plateMass, actuatorTop, actuatorBottom, grav = 9.81, tPlateMass = 0):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SetMasses","setMasses")
        return self.setMasses(plateMass, actuatorTop, actuatorBottom, grav, tPlateMass)
    def SetGrav(self, grav = 9.81):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SetGrav","setGrav")
        return self.setGrav(grav)
    def SetCOG(self, motor_grav_center, shaft_grav_center):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SetCOG","setCOG")
        return self.setCOG(motor_grav_center, shaft_grav_center)
    def SetAngleDev(self, MaxAngleDev = 55):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SetAngleDev","setMaxAngleDev")
        return self.setMaxAngleDev(MaxAngleDev)
    def SetPlateAngleDev(self, MaxPlateDev = 60):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SetPlateAngleDev","setMaxPlateRotation")
        return self.setMaxPlateRotation(MaxPlateDev)
    def SetDrawingDimensions(self, OuterTopRad, OuterBotRad, ShaftRad, MotorRad):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SetDrawingDimensions","setDrawingDimensions")
        return self.setDrawingDimensions( OuterTopRad, OuterBotRad, ShaftRad, MotorRad)
    def _setPlatePos(self, bottomT, topT):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("_setPlatePos","setPlatePos")
        return self.setPlatePos(bottomT, topT)
    def gLens(self):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("gLens","getLens")
        return self.getLens()
    def gtopT(self):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("gtopT","getTopT")
        return self.getTopT()
    def gbottomT(self):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("gbottomT","getBottomT")
        return self.getBottomT()
    def GetActuatorUnit(self, p1, p2, dist):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("GetActuatorUnit","getActuatorUnitVec")
        return self.getActuatorUnitVec(p1, p2, dist)
    def SpinCustom(self, rot):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SpinCustom","spinCustom")
        return self.spinCustom(rot)
    def SimplifiedRaphson(self, L, bottomT = None, reverse = False, protect = False):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SimplifiedRaphson","simplifiedRaphson")
        return self.simplifiedRaphson(L, bottomT, reverse, protect)
    def LambdaRTP(self, stopt):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("LambdaRTP","lambdaTopPlateReorientation")
        return self.lambdaTopPlateReorientation(stopt)
    def ReorientTopPlate(self):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("ReorientTopPlate","reorientTopPlate")
        return self.reorientTopPlate()
    def _legLengthConstraint(self, donothing):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("_legLengthConstraint","legLengthConstraint")
        return self.legLengthConstraint(donothing)
    def _resclLegs(self, cMin, cMax):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("_resclLegs","rescaleLegLengths")
        return self.rescaleLegLengths(cMin, cMax)
    def _addLegs(self, cMin, cMax):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("_addLegs","addLegsToMinimum")
        return self.addLegsToMinimum(cMin, cMax)
    def _subLegs(self, cMin, cMax):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("_subLegs","subLegsToMaximum")
        return self.subLegsToMaximum(cMin, cMax)
    def _lengthCorrectiveAction(self):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("_lengthCorrectiveAction","lengthCorrectiveAction")
        return self.lengthCorrectiveAction()
    def _continuousTranslationConstraint(self):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction(
            "_continuousTranslationConstraint","continuousTranslationConstraint")
        return self.continuousTranslationConstraint()
    def _continuousTranslationCorrectiveAction(self):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction(
            "_continuousTranslationCorrectiveAction","continuousTranslationCorrectiveAction")
        return self.continuousTranslationCorrectiveAction()
    def _interiorAnglesConstraint(self):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("_interiorAnglesConstraint","interiorAnglesConstraint")
        return self.interiorAnglesConstraint()
    def AngleFromNorm(self):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("AngleFromNorm","getJointAnglesFromNorm")
        return self.getJointAnglesFromNorm()
    def AngleFromVertical(self):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("AngleFromVertical","getJointAnglesFromVertical")
        return self.getJointAnglesFromVertical()
    def _bottomTopCheck(self, bottomT, topT):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("_bottomTopCheck","bottomTopCheck")
        return self.bottomTopCheck(bottomT, topT)
    def JacobianSpace(self, bottomT = None, topT = None):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("JacobianSpace","jacobianSpace")
        return self.jacobianSpace(bottomT, topT)
    def InverseJacobianSpace(self, bottomT = None, topT = None, protect = True):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("InverseJacobianSpace","inverseJacobianSpace")
        return self.inverseJacobianSpace(bottomT, topT)
    def AltInverseJacobianSpace(self, bottomT = None, topT = None, protect = True):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("AltInverseJacobianSpace","altInverseJacobianSpace")
        return self.altInverseJacobianSpace(bottomT, topT, protect)
    def CarryMassCalc(self, twrench, protect = False):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("CarryMassCalc","carryMassCalc")
        return self.carryMassCalc(twrench, protect)
    def CarryMassCalcNew(self, twrench, protect = False):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("CarryMassCalcNew","carryMassCalcLocal")
        return self.carryMassCalcLocal(twrench, protect)
    def MeasureForcesAtEENew(self, wrench, protect = False):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("MeasureForcesAtEENew","measureForcesAtEENew")
        return self.measureForcesAtEENew(wrench, protect)
    def CarryMassCalcUp(self, twrench, protect = False):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("CarryMassCalcUp","carryMassCalcUp")
        return self.carryMassCalcUp(twrench, protect)
    def MeasureForcesFromWrenchEE(self, bottomT = np.zeros((1)) ,
        topT = np.zeros((1)), topWEE = np.zeros((1)), protect = True):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("MeasureForcesFromWrenchEE","measureForcesFromWrenchEE")
        return self.measureForcesFromWrenchEE(bottomT, topT, topWEE, protect)
    def MeasureForcesFromBottomEE(self, bottomT = np.zeros((1)) ,
        topT = np.zeros((1)), topWEE = np.zeros((1)), protect = True):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("MeasureForcesFromBottomEE","measureForcesFromBottomEE")
        return self.measureForcesFromBottomEE(bottomT, topT, topWEE, protect)
    def WrenchEEFromMeasuredForces(self, bottomT, topT, tau):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("WrenchEEFromMeasuredForces","wrenchEEFromMeasuredForces")
        return self.wrenchEEFromMeasuredForces(bottomT, topT, tau)
    def WrenchBottomFromMeasuredForces(self, bottomT, topT, tau):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction(
            "WrenchBottomFromMeasuredForces","wrenchBottomFromMeasuredForces")
        return self.wrenchBottomFromMeasuredForces(bottomT, topT, tau)
    def SumActuatorWrenches(self, forces = None):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SumActuatorWrenches","sumActuatorWrenches")
        return self.sumActuatorWrenches(forces)

def loadSP(fname, file_directory = "../robot_definitions/", baseloc = None, altRot = 1):
    """
    Loads A Stewart Platform Object froma  file

    Args:
        fname (String): Description of parameter `fname`.
        file_directory (String): Description of parameter `file_directory`.
        baseloc (tm): Description of parameter `baseloc`.
        altRot (Float): Description of parameter `altRot`.

    Returns:
        type: Description of returned object.

    """
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
    outer_top_radius = sp_data["Drawing"]["TopRadius"]
    outer_bottom_radius = sp_data["Drawing"]["BottomRadius"]
    act_shaft_radius = sp_data["Drawing"]["ShaftRadius"]
    act_motor_radius = sp_data["Drawing"]["MotorRadius"]
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


    newsp = newSP(bot_radius, top_radius, bot_joint_spacing, top_joint_spacing,
        bot_thickness, top_thickness, actuator_shaft_mass, actuator_motor_mass, plate_top_mass,
        plate_bot_mass, motor_grav_center, shaft_grav_center,
        actuator_min, actuator_max, baseloc, name, altRot)

    newsp.setDrawingDimensions(
        outer_top_radius,
        outer_bottom_radius,
        act_shaft_radius,
        act_motor_radius)
    newsp.setMaxAngleDev(max_dev)
    newsp.force_limit = force_lim

    return newsp
def newSP(bottom_radius, top_radius, bJointSpace, tJointSpace,
    bottom_plate_thickness, top_plate_thickness, actuator_shaft_mass,
    actuator_motor_mass, plate_top_mass, plate_bot_mass, motor_grav_center,
    shaft_grav_center, actuator_min, actuator_max, base_location, name, rot = 1):
    """

    Args:
        bottom_radius (Float): Description of parameter `bottom_radius`.
        top_radius (Float): Description of parameter `top_radius`.
        bJointSpace (ndarray(Float)): Description of parameter `bJointSpace`.
        tJointSpace (ndarray(Float)): Description of parameter `tJointSpace`.
        bottom_plate_thickness (Float): Description of parameter `bottom_plate_thickness`.
        top_plate_thickness (Float): Description of parameter `top_plate_thickness`.
        actuator_shaft_mass (Float): Description of parameter `actuator_shaft_mass`.
        actuator_motor_mass (Float): Description of parameter `actuator_motor_mass`.
        plate_top_mass (Float): Description of parameter `plate_top_mass`.
        plate_bot_mass (Float): Description of parameter `plate_bot_mass`.
        motor_grav_center (Float): Description of parameter `motor_grav_center`.
        shaft_grav_center (Float): Description of parameter `shaft_grav_center`.
        actuator_min (Float): Description of parameter `actuator_min`.
        actuator_max (Float): Description of parameter `actuator_max`.
        base_location (tm): Description of parameter `base_location`.
        name (String): Description of parameter `name`.
        rot (Float): Description of parameter `rot`.

    Returns:
        type: Description of returned object.

    """

    bottom_gap = bJointSpace / 2 * np.pi / 180
    top_gap = tJointSpace / 2 * np.pi / 180

    bottom_joint_gap = 120 * np.pi / 180 #Angle of seperation between joint clusters
    top_joint_gap = 60 * np.pi / 180 #Offset in rotation of the top plate versus the bottom plate

    bangles =  np.array([
        -bottom_gap, bottom_gap,
        bottom_joint_gap-bottom_gap,
        bottom_joint_gap+bottom_gap,
        2*bottom_joint_gap-bottom_gap,
        2*bottom_joint_gap+bottom_gap])
    tangles = np.array([
        -top_joint_gap+top_gap,
        top_joint_gap-top_gap,
        top_joint_gap+top_gap,
        top_joint_gap+bottom_joint_gap-top_gap,
        top_joint_gap+bottom_joint_gap+top_gap,
        -top_joint_gap-top_gap])
    if rot == -1:
        tangles =  np.array([
            -bottom_gap, bottom_gap,
            bottom_joint_gap-bottom_gap,
            bottom_joint_gap+bottom_gap,
            2*bottom_joint_gap-bottom_gap,
            2*bottom_joint_gap+bottom_gap])
        bangles = np.array([
            -top_joint_gap+top_gap,
            top_joint_gap-top_gap,
            top_joint_gap+top_gap,
            top_joint_gap+bottom_joint_gap-top_gap,
            top_joint_gap+bottom_joint_gap+top_gap,
            -top_joint_gap-top_gap])

    S = fmr.ScrewToAxis(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 0).reshape((6, 1))

    Mb = tm(np.array([bottom_radius, 0.0, 0.0, 0.0, 0.0, 0.0]))
     #how far from the bottom plate origin should clusters be generated
    Mt = tm(np.array([top_radius, 0.0, 0.0, 0.0, 0.0, 0.0]))
     #Same thing for the top

    bj = np.zeros((3, 6)) #Pre allocate arrays
    tj = np.zeros((3, 6))

    for i in range(0, 6):
        bji = fsr.TransformFromTwist(bangles[i] * S) @ Mb
        tji = fsr.TransformFromTwist(tangles[i] * S) @ Mt
        bj[0:3, i] = bji[0:3].reshape((3))
        tj[0:3, i] = tji[0:3].reshape((3))
        bj[2, i] = bottom_plate_thickness
        tj[2, i] = -top_plate_thickness

    bottom = base_location.copy()
    tentative_height = midHeightEstimate(
        actuator_min, actuator_max, bj, bottom_plate_thickness, top_plate_thickness)
    if rot == -1:
        tentative_height = midHeightEstimate(
            actuator_min, actuator_max, tj, bottom_plate_thickness, top_plate_thickness)
    top = bottom @ tm(np.array([0.0, 0.0, tentative_height, 0.0, 0.0, 0.0]))

    newsp = SP(bj, tj, bottom, top,
        actuator_min, actuator_max,
        bottom_plate_thickness, top_plate_thickness, name)
    newsp.setMasses(
        plate_bot_mass,
        actuator_shaft_mass,
        actuator_motor_mass,
        top_plate_mass = plate_top_mass)
    newsp.setCOG(motor_grav_center, shaft_grav_center)

    return newsp
def makeSP(bRad, tRad, spacing, baseT,
    platOffset, rot = -1, plate_thickness_avg = 0, altRot = 0):
    """
    Builds a new stewart platform object.

    Args:
        bRad (Float): Description of parameter `bRad`.
        tRad (Float): Description of parameter `tRad`.
        spacing (Float): Description of parameter `spacing`.
        baseT (tm): Description of parameter `baseT`.
        platOffset (Float): Description of parameter `platOffset`.
        rot (Float): Description of parameter `rot`.
        plate_thickness_avg (Float): Description of parameter `plate_thickness_avg`.
        altRot (Float): Description of parameter `altRot`.

    Returns:
        type: Description of returned object.

    """
    gapS = spacing/2*np.pi/180 #Angle between cluster joints
    bottom_joint_gap = 120*np.pi/180 #Angle of seperation between joint clusters
    top_joint_gap = 60*np.pi/180 #Offset in rotation of the top plate versus the bottom plate
    bangles = np.array([
        -gapS,
        gapS,
        bottom_joint_gap-gapS,
        bottom_joint_gap+gapS,
        2*bottom_joint_gap-gapS,
        2*bottom_joint_gap+gapS]) + altRot*np.pi/180
    tangles = np.array([
        -top_joint_gap+gapS,
        top_joint_gap-gapS,
        top_joint_gap+gapS,
        top_joint_gap+bottom_joint_gap-gapS,
        top_joint_gap+bottom_joint_gap+gapS,
        -top_joint_gap-gapS])+ altRot*np.pi/180
    if rot == -1:
        tangles = np.array([
            -gapS, gapS,
            bottom_joint_gap-gapS,
            bottom_joint_gap+gapS,
            2*bottom_joint_gap-gapS,
            2*bottom_joint_gap+gapS])+ altRot*np.pi/180
        bangles = np.array([
            -top_joint_gap+gapS,
            top_joint_gap-gapS,
            top_joint_gap+gapS,
            top_joint_gap+bottom_joint_gap-gapS,
            top_joint_gap+bottom_joint_gap+gapS,
            -top_joint_gap-gapS])+ altRot*np.pi/180

    disp(bangles, "bangles")
    disp(tangles, "tangles")
    S = fmr.ScrewToAxis(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 0).reshape((6, 1))

    Mb = tm(np.array([bRad, 0.0, 0.0, 0.0, 0.0, 0.0]))
     #how far from the bottom plate origin should clusters be generated
    Mt = tm(np.array([tRad, 0.0, 0.0, 0.0, 0.0, 0.0]))
     #Same thing for the top

    bj = np.zeros((3, 6)) #Pre allocate arrays
    tj = np.zeros((3, 6))

    #Generate position vectors (XYZ) for top and bottom joint locations
    for i in range(0, 6):
        bji = fsr.TransformFromTwist(bangles[i] * S) @ Mb
        tji = fsr.TransformFromTwist(tangles[i] * S) @ Mt
        bj[0:3, i] = bji[0:3].reshape((3))
        tj[0:3, i] = tji[0:3].reshape((3))
        bj[2, i] = plate_thickness_avg/2
        tj[2, i] = -plate_thickness_avg/2

    #if rot == -1:
    #    disp(bj, "Prechange")
#
#        rotby = TAAtoTM(np.array([0, 0, 0, 0, 0, np.pi/3]))
#        for i in range(6):
#            bj[0:3, i] = TMtoTAA(rotby @
#                TAAtoTM(np.array([bj[0, i], bj[1, i], bj[2, i], 0, 0, 0])))[0:3].reshape((3))
#            tj[0:3, i] = TMtoTAA(rotby @
#                TAAtoTM(np.array([tj[0, i], tj[1, i], tj[2, i], 0, 0, 0])))[0:3].reshape((3))
#        disp(bj, "postchange")
    bottom = baseT.copy()
    #Generate top position at offset from the bottom position
    top = bottom @ tm(np.array([0.0, 0.0, platOffset, 0.0, 0.0, 0.0]))
    sp = SP(bj, tj, bottom, top, 0, 0, plate_thickness_avg, plate_thickness_avg, 'sp')
    sp.bRad = bRad
    sp.tRad = tRad

    return sp, bottom, top
#Helpers
def midHeightEstimate(leg_ext_min, leg_ext_max, bj, bth, tth):
    """

    Args:
        leg_ext_min (float): Description of parameter `leg_ext_min`.
        leg_ext_max (float): Description of parameter `leg_ext_max`.
        bj (array(float)): Description of parameter `bj`.
        bth (tm): Description of parameter `bth`.
        tth (tm): Description of parameter `tth`.

    Returns:
        type: Description of returned object.

    """
    s1 = (leg_ext_min + leg_ext_max) / 2
    d1 = fsr.Distance(tm([bj[0, 0], bj[1, 0], bj[2, 0], 0, 0, 0]),
            tm([bj[0, 1], bj[1, 1], bj[2, 1], 0, 0, 0]))
    hest = (np.sqrt(s1 ** 2 - d1 **2)) + bth + tth
    return hest
