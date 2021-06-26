from faser_math import tm, fmr, fsr
from faser_plotting.Draw.Draw import DrawArm, DrawRectangle
from faser_utils.disp.disp import disp
import numpy as np
import scipy as sci
import scipy.linalg as ling
import random
from os.path import dirname, basename, isfile

class Arm:
    #Conventions:
    #Filenames:  snake_case
    #Variables: snake_case
    #Functions: camelCase
    #ClassNames: CapsCase
    #Docstring: Google

    #Converted to python - Liam
    def __init__(self, base_pos_global, screw_list, end_effector_home,
            joint_poses_home, joint_axes = np.array([0])):
        """"
        Create a serial arm
        Args:
            base_pos_global: Base transform of Arm. tmobject
            screw_list: Screw list of arm. Nx6 matrix.
            end_effector_home: Initial end effector of arm.
            joint_poses_home: joint_poses_home list for arm
            joint_axes: joint_axes list for arm
        """
        #Configure the arm given the base screws, the base transform.
        self.cameras = []
        self.initialize(base_pos_global, screw_list, end_effector_home, joint_poses_home)
        self.link_mass_transforms = 0
        self.box_spatial_links = 0
        self.link_home_positions = None
        self.link_dimensions = None
        self.fail_count = 0
        for i in range(0, screw_list.shape[1]):
            self.screw_list_body[:, i] = (fmr.Adjoint(self.end_effector_home.inv().gTM()) @
                self.screw_list[:, i])
        self.reversable = False
        if joint_axes.shape[0] != 1:
            self.reversable = True
            self.reversed = False
            self.joint_axes = joint_axes
            self.original_joint_axes = joint_axes
        self.original_screw_list = self.screw_list
        self.joint_mins = np.ones((screw_list.shape[1])) * np.pi * -1
        self.joint_maxs = np.ones((screw_list.shape[1])) * np.pi
        self.FK(np.zeros((self.screw_list.shape[1])))

    def initialize(self, base_pos_global, screw_list, end_effector_home, joint_poses_home):
        """
        Helper for Serial Arm. Should be called internally
        Args:
            base_pos_global: Base transform of Arm. tmobject
            screw_list: Screw list of arm. Nx6 matrix.
            end_effector_home: Initial end effector of arm.
            joint_poses_home: joint_poses_home list for arm
        """
        self.screw_list = screw_list
        self.original_screw_list_body = np.copy(screw_list)
        self.base_pos_global = base_pos_global
        self.original_joint_poses_home = joint_poses_home
        self.joint_poses_home = np.zeros((3, screw_list.shape[1]))
        self.screw_list_body = np.zeros((6, screw_list.shape[1]))
        if joint_poses_home.size > 1:
            for i in range(0, screw_list.shape[1]):
                self.joint_poses_home[0:3, i] = fsr.TrVec(base_pos_global, joint_poses_home[0:3, i])
                #Convert TrVec
        for i in range(0, screw_list.shape[1]):
            self.screw_list[:, i] = fmr.Adjoint(base_pos_global.gTM()) @ screw_list[:, i]
            if joint_poses_home.size <= 1:
                [w, th, joint_pose_temp, h] = fmr.TwistToScrew(self.screw_list[:, i])
                #Convert TwistToScrew
                self.joint_poses_home[0:3, i] = joint_pose_temp; # For plotting purposes
        self.end_effector_home_local = end_effector_home
        self.end_effector_home = base_pos_global @ end_effector_home
        self.end_effector_pos_global = self.end_effector_home.copy()
        self.original_end_effector_home = self.end_effector_home.copy()

    """
    Compatibility, for those to be deprecated
    """
    def printOutOfDateFunction(sef, old_name, use_name):
        print(old_name + " is deprecated. Please use " + use_name + " instead.")
    def RandomPos(self):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("RandomPos", "randomPos")
        return self.randomPos()
    def Reverse(self):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("Reverse", "reverse")
        self.reverse()
    def vServoSP(self, target, tol = 2, ax = 0, plt = 0, fig = 0):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("vServoSP","visualServoToTarget")
        return self.visualServoToTarget(target, tol, ax, plt, fig)
    def SetDynamicsProperties(self, _Mlinks = None, _Mhome = None, _Glinks = None, _Dims = None):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SetDynamicsProperties", "setDynamicsProperties")
        return self.setDynamicsProperties(_Mlinks, _Mhome, _Glinks, _Dims)
    def SetMasses(self, mass):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SetMasses","setMasses")
        return self.setMasses(mass)
    def TestArmValues(self):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("TestArmValues","testArmValues")
        return self.testArmValues()
    def SetArbitraryHome(self, theta,T):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SetArbitraryHome","setArbitraryHome")
        return self.setArbitraryHome(theta, T)
    def RestoreOriginalEE(self):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("RestoreOriginalEE","restoreOriginalEE")
        return self.restoreOriginalEE()
    def StaticForces(self, theta, wrenchEE):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("StaticForces","staticForces")
        return self.staticForces(theta, wrenchEE)
    def StaticForcesInv(self, theta, tau):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("StaticForcesInv","staticForcesInv")
        return self.staticForcesInv(theta, tau)
    def InverseDynamics(self, theta, thetadot, thetadotdot, grav, wrenchEE):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("InverseDynamics","inverseDynamics")
        return self.inverseDynamics(theta, thetadot, thetadotdot, grav, wrenchEE)
    def InverseDynamicsEMR(self, theta, thetadot, thetadotdot, grav, wrenchEE):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("InverseDynamicsEMR","inverseDynamicsEMR")
        return self.inverseDynamicsEMR(theta, thetadot, thetadotdot, grav, wrenchEE)
    def InverseDynamicsE(self, theta, thetadot, thetadotdot, grav, wrenchEE):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("InverseDynamicsE","inverseDynamicsE")
        return self.inverseDynamicsE(theta, thetadot, thetadotdot, grav, wrenchEE)
    def InverseDynamicsC(self, theta, thetadot, thetadotdot, grav, wrenchEE):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("InverseDynamicsC","inverseDynamicsC")
        return self.inverseDynamicsC(theta, thetadot, thetadotdot, grav, wrenchEE)
    def ForwardDynamicsE(self, theta, thetadot, tau, grav, wrenchEE):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("ForwardDynamicsE","forwardDynamicsE")
        return self.forwardDynamicsE(theta, thetadot, tau, grav, wrenchEE)
    def ForwardDynamics(self, theta, thetadot, tau, grav, wrenchEE):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("ForwardDynamics","forwardDynamics")
        return self.forwardDynamics(theta, thetadot, tau, grav, wrenchEE)
    def MassMatrix(self, theta):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("MassMatrix","massMatrix")
        return self.massMatrix(theta)
    def CoriolisGravity(self, theta, thetadot, grav):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("CoriolisGravity","coriolisGravity")
        return self.coriolisGravity(theta, thetadot, grav)
    def EndEffectorForces(self, theta, wrenchEE):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("EndEffectorForces","endEffectorForces")
        return self.endEffectorForces(theta, wrenchEE)
    def Jacobian(self, theta):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("Jacobian","jacobian")
        return self.jacobian(theta)
    def JacobianBody(self, theta):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("JacobianBody","jacobianBody")
        return self.jacobianBody(theta)
    def JacobianLink(self, theta, i):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("JacobianLink","jacobianLink")
        return self.jacobianLink(theta, i)
    def JacobianEE(self, theta):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("JacobianEE","jacobianEE")
        return self.jacobianEE(theta)
    def JacobianEEtrans(self, theta):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("JacobianEEtrans","jacobianEEtrans")
        return self.jacobianEEtrans(theta)
    def NumericalJacobian(self, theta):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("NumericalJacobian","numericalJacobian")
        return self.numericalJacobian(theta)
    def GetManipulability(self, theta = None):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("GetManipulability","getManipulability")
        return self.getManipulability(theta)
    def Draw(self, ax):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("Draw","draw")
        return self.draw(ax)


    """
       _  ___                            _   _
      | |/ (_)                          | | (_)
      | ' / _ _ __   ___ _ __ ___   __ _| |_ _  ___ ___
      |  < | | '_ \ / _ \ '_ ` _ \ / _` | __| |/ __/ __|
      | . \| | | | |  __/ | | | | | (_| | |_| | (__\__ \
      |_|\_\_|_| |_|\___|_| |_| |_|\__,_|\__|_|\___|___/
    """
    def thetaProtector(self, theta):
        """
        Properly bounds theta values
        Args:
            theta: joint angles to be tested and reset
        Returns:
            newtheta: corrected joint angles
        """
        theta[np.where(theta<self.joint_mins)] = self.joint_mins[np.where(theta<self.joint_mins)]
        theta[np.where(theta>self.joint_maxs)] = self.joint_maxs[np.where(theta>self.joint_maxs)]
        return theta

    #Converted to python -Liam
    def FK(self, theta, protect = False):
        """
        Calculates the end effector position of the serial arm given thetas
        params:
            theta: input joint array
            protect: whether or not to validate action
        returns:
            end_effector_transform: End effector tm
        """
        if not protect and (np.any(theta < self.joint_mins) or np.any(theta > self.joint_maxs)):
            theta = self.thetaProtector(theta)
        self._theta = fsr.AngleMod(theta.reshape(len(theta)))
        end_effector_transform = tm(fmr.FKinSpace(
            self.end_effector_home.gTM(), self.screw_list, theta))
        self.end_effector_pos_global = end_effector_transform
        return end_effector_transform

    #Converted to python - Liam
    def FKLink(self, theta, i, protect = False):
        """
        Calculates the position of a given joint provided a theta list
        Args:
            theta: The array of theta values for each joint
            i: The index of the joint desired, from 0
        """
        # Returns the TM of link i
        # Lynch 4.1
        if not protect and (np.any(theta < self.joint_mins) or np.any(theta > self.joint_maxs)):
            print("Unsuitable Thetas")
            theta = self.thetaProtector(theta)
        end_effector_pos =  tm(fmr.FKinSpace(self.link_home_positions[i].TM,
            self.screw_list[0:6, 0:i], theta[0:i]))
        return end_effector_pos

    #Converted to python - Liam
    def IK(self,T, theta_init = np.zeros(1), check = 1, level = 6, protect = False):
        """
        Calculates joint positions of a serial arm. All parameters are
            optional except the desired end effector position
        Args:
            T: Desired end effector position to calculate for
            theta_init: Intial theta guess for desired end effector position.
                Set to 0s if not provided.
            check: Whether or not the program should retry if the position finding fails
            level: number of recursive calls allowed if check is enabled
        Returns:
            List of thetas, success boolean
        """
        if theta_init.size == 1:
            theta_init = fsr.AngleMod(self._theta.reshape(len(self._theta)))
        if not protect:
            return self.constrainedIK(T, theta_init, check, level)
        theta, success = fmr.IKinSpace(self.screw_list, self.end_effector_home.gTM(),
            T.gTM(), theta_init, 0.00000001, 0.00000001)
        theta = fsr.AngleMod(theta)
        self._theta = theta
        if success:
            self.end_effector_pos_global = T
        else:
            if check == 1:
                i = 0
                while i < level and success == 0:
                    theta_temp = np.zeros((len(self._theta)))
                    for j in range(len(theta_temp)):
                        theta_temp[j] = random.uniform(-np.pi, np.pi)
                    theta, success = fmr.IKinSpace(self.screw_list, self.end_effector_home.gTM(),
                        T.gTM(), theta_init, 0.00000001, 0.00000001)
                    i = i + 1
                if success:
                    self.end_effector_pos_global = T
        return theta, success


    def constrainedIK(self, T, theta_init, check = 1, level = 6):
        """
        Calculates joint positions of a serial arm, provided rotational constraints on the Joints
        All parameters are optional except the desired end effector position
        Joint constraints are set through the joint_maxs and joint_mins properties, and should be
        arrays the same size as the number of DOFS
        Args:
            T: Desired end effector position to calculate for
            theta_init: Intial theta guess for desired end effector position.
                Set to 0s if not provided.
            check: Whether or not the program should retry if the position finding fails
            level: number of recursive calls allowed if check is enabled
        Returns:
            List of thetas, success boolean
        """
        if not isinstance(T, tm):
            print(T)
            print("Attempted pass ^")
            return self._theta
        screw_list = self.screw_list.copy()
        if check == 1:
            self.fail_count = 0
        M = self.end_effector_home.copy()
        pos_tolerance = .001
        rot_tolerance = .0001
        theta_list = self._theta.copy()
        i = 0
        max_iterations = 30

        try:
            theta_list, success = fmr.IKinSpaceConstrained(screw_list, M.gTM(),
                T.gTM(), theta_list, pos_tolerance, rot_tolerance,
                self.joint_mins, self.joint_maxs, max_iterations)
        except:
            theta_list, success = self.constrainedIKNoFMR(screw_list, M, T, theta_list,
                pos_tolerance, rot_tolerance, max_iterations)
        if success:
            self.end_effector_pos_global = T
        else:
            if check == 1:
                i = 0
                while i < level and success == 0:
                    theta_temp = np.zeros((len(self._theta)))
                    for j in range(len(theta_temp)):
                        theta_temp[j] = random.uniform(self.joint_mins[j], self.joint_maxs[j])
                    try:
                        theta_list, success = fmr.IKinSpaceConstrained(screw_list, M.gTM(),
                            T.gTM(), theta_temp, pos_tolerance, rot_tolerance,
                            self.joint_mins, self.joint_maxs, max_iterations)
                    except Exception as e:
                        theta_list, success = self.constrainedIK(T, theta_temp, check = 0)
                        disp("FMR Failure: " + str(e))
                    i = i + 1
                if success:
                    self.end_effector_pos_global = T
        if not success:
            if check == 0:
                self.fail_count += 1
            else:
                #print("Total Cycle Failure")
                self.FK(np.zeros(len(self._theta)))
        else:
            if self.fail_count != 0:
                print("Success + " + str(self.fail_count) + " failures")
            self.FK(theta_list)

        return theta_list, success

    def constrainedIKNoFMR(self,
        screw_list, M, T, theta_list, pos_tolerance, rot_tolerance, max_iterations):
        """
        Used as a backup function for the standard constrained IK
        Args:
            screw_list: screw list
            M: home end effector position
            T: Goal Position
            theta_list: Initial thetas
            pos_tolerance: Positional tolerance
            rot_tolerance: Rotational tolerance
            max_iterations: Maximum Iterations before failure
        Returns:
            theta_list: list of solved thetas
            success: boolean of success
        """
        end_effector_pos_temp = fmr.FKinSpace(M.gTM(), screw_list, theta_list)
        error_vec = np.dot(fmr.Adjoint(end_effector_pos_temp), fmr.se3ToVec(
            fmr.MatrixLog6(np.dot(fmr.TransInv(end_effector_pos_temp), T.gTM()))))
        #print(fmr.MatrixLog6(np.dot(fmr.TransInv(end_effector_pos_temp), T)), "Test")
        err = np.linalg.norm([error_vec[0], error_vec[1], error_vec[2]]) > pos_tolerance \
              or np.linalg.norm([error_vec[3], error_vec[4], error_vec[5]]) > rot_tolerance
        if np.isnan(error_vec).any():
            err = True
        i = 0
        while err and i < max_iterations:
            theta_list = theta_list \
                        + np.dot(np.linalg.pinv(fmr.JacobianSpace(
                        screw_list, theta_list)), error_vec)
            for j in range(len(theta_list)):
                if theta_list[j] < self.joint_mins[j]:
                    theta_list[j] = self.joint_mins[j]
                if theta_list[j] > self.joint_maxs[j]:
                    theta_list[j] = self.joint_maxs[j];
            i = i + 1
            end_effector_pos_temp = fmr.FKinSpace(M.gTM(), screw_list, theta_list)
            error_vec = np.dot(fmr.Adjoint(end_effector_pos_temp), \
                        fmr.se3ToVec(fmr.MatrixLog6(np.dot(
                        fmr.TransInv(end_effector_pos_temp), T.gTM()))))
            err = np.linalg.norm([error_vec[0], error_vec[1], error_vec[2]]) > pos_tolerance \
                  or np.linalg.norm([error_vec[3], error_vec[4], error_vec[5]]) > rot_tolerance
            if np.isnan(error_vec).any():
                err = True
        success = not err
        return theta_list, success

    def IKForceOptimal(self, T, theta_init, forcev, random_sample = 1000, mode = "MAX"):
        """
        Early attempt at creating a force optimization package for a serial arm.
        Absolutely NOT the optimial way to do this. Only works for overactuated arms.
        Args:
            T: Desired end effector position to calculate for
            theta_init: Intial theta guess for desired end effector position.
                Set to 0s if not provided.
            forcev: Force applied to the end effector of the arm. Wrench.
            random_sample: number of samples to test to look for the most optimal solution
            mode: Set Mode to reduce. Max: Max Force. Sum: Sum of forces. Mean: Mean Force
        Returns:
            List of thetas
        """
        thetas = []
        for i in range(random_sample):
            theta_temp = np.zeros((len(self._theta)))
            for j in range(len(theta_init)):
                theta_temp[j] = random.uniform(-np.pi, np.pi)
            thetas.append(theta_temp)
        force_thetas = []
        temp_moment = np.cross(T[0:3].reshape((3)), forcev)
        wrench = np.array([temp_moment[0], temp_moment[1],
            temp_moment[2], forcev[0], forcev[1], forcev[2]]).reshape((6, 1))
        for i in range(len(thetas)):
            candidate_theta, success =self.IK(T, thetas[i])
            if success and sum(abs(fsr.Error(self.FK(candidate_theta), T))) < .0001:
                force_thetas.append(candidate_theta)
        max_force = []
        for i in range(len(force_thetas)):
            if mode == "MAX":
                max_force.append(max(abs(self.staticForces(force_thetas[i], wrench))))
            elif mode == "SUM":
                max_force.append(sum(abs(self.staticForces(force_thetas[i], wrench))))
            elif mode == "MEAN":
                max_force.append(sum(abs(
                    self.staticForces(force_thetas[i], wrench))) / len(force_thetas))
        index = max_force.index(min(max_force))
        self._theta = force_thetas[index]
        return force_thetas[index]

    def IKMotion(self, T, theta_init):
        """
        This calculates IK by numerically moving the end effector
        from the pose defined by theta_init in the direction of the desired
        pose T.  If the pose cannot be reached, it gets as close as
        it can.  This can sometimes return a better result than IK.
        An approximate explanation an be found in Lynch 9.2
        Args:
            T: Desired end effector position to calculate for
            theta_init: Intial theta guess for desired end effector position.
        Returns:
            theta: list of theta lists
            success: boolean for success
            t: goal
            thall: integration results
        """
        start_transform = self.FK(theta_init)
        start_direction = T @ start_transform.inv()
        twist_direction = fsr.TwistFromTransform(start_direction)

        #[t, thall] = ode15s(@(t, x)(pinv(self.jacobian(x))*twist_direction),[0 1], theta_init);
        res = lambda t, x: np.linalg.pinv(self.jacobian(x))*twist_direction
        t, thall = sci.integrate.ode(res).set_integrator('vode', method='bdf', order=15)

        theta = thall[-1,:].conj().T
        if fsr.Norm(T-self.FK(theta)) < 0.001:
            success = 1
        else:
            success = 0
        return theta, success, t, thall

    def IKFree(self,T, theta_init, inds):
        """
        Only allow theta_init(freeinds) to be varied
        Method not covered in Lynch.
        SetElements inserts the variable vector x into the positions
        indicated by freeinds in theta_init.  The remaining elements are
        unchanged.
        Args:
            T: Desired end effector position to calculate for
            theta_init: Intial theta guess for desired end effector position.
            inds: Free indexes to move
        Returns:
            theta: list of theta lists
            success: boolean for success
            t: goal
            thall: integration results
        """
        #free_thetas = fsolve(@(x)(obj.FK(SetElements(theta_init,
            #freeinds, x))-T), theta_init(freeinds))
        res = lambda x : fsr.TAAtoTM(self.FK(fsr.SetElements(theta_init, inds, x))-T)
        #Use newton_krylov instead of fsolve
        free_thetas = sci.optimize.fsolve(res, theta_init[inds])
        # free_thetas = fsolve(@(x)(self.FK(SetElements(theta_init,
            #freeinds, x))-T), theta_init(freeinds));
        theta = np.squeeze(theta_init);
        theta[inds] = np.squeeze(free_thetas);
        if fsr.Norm(T-self.FK(theta)) < 0.001:
            success = 1;
        else:
            success = 0;
        return (theta, success)


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

    def randomPos(self):
        """
        Create a random position, return the end effector TF
        Returns:
            random pos
        """
        theta_temp = np.zeros((len(self._theta)))
        for j in range(len(theta_temp)):
            theta_temp[j] = random.uniform(self.joint_mins[j], self.joint_maxs[j])
        pos = self.FK(theta_temp)
        return pos


    def reverse(self):
        """
        Flip around the serial arm so that the end effector is not the base and vice versa.
        Keep the same end pose
        """
        if not self.reversable:
            return
        old_thetas = np.copy(self._theta)
        new_theta = np.zeros((len(self._theta)))
        for i in range(self.screw_list.shape[1]):
            new_theta[i] = old_thetas[len(old_thetas) - 1 - i]
        new_screw_list = np.copy(self.original_screw_list)
        new_end_effector_home = self.end_effector_home.copy()
        new_thetas = self.FK(self._theta)
        new_joint_axes = np.copy(self.joint_axes)
        new_joint_poses_home = np.copy(self.original_joint_poses_home)
        for i in range(new_joint_axes.shape[1]):
            new_joint_axes[0:3, i] = self.joint_axes[0:3, new_joint_axes.shape[1] - 1 - i]
        differences = np.zeros((3, new_joint_poses_home.shape[1]-1))
        for i in range(new_joint_poses_home.shape[1]-1):
            differences[0:3, i] = (self.original_joint_poses_home[0:3,(
                self.original_joint_poses_home.shape[1] - 1 - i)] -
                self.original_joint_poses_home[0:3,(
                self.original_joint_poses_home.shape[1] - 2 - i)])
        #print(differences, "differences")
        for i in range(new_joint_poses_home.shape[1]):
            if i == 0:
                new_joint_poses_home[0:3, i] = (self.original_joint_poses_home[0:3, (
                    self.original_joint_poses_home.shape[1] - 1)] - np.sum(differences, axis = 1))
            else:
                new_joint_poses_home[0:3, i] = (new_joint_poses_home[0:3, i -1] +
                    differences[0:3, i - 1])
        for i in range(new_screw_list.shape[1]):
            new_screw_list[0:6, i] = np.hstack((new_joint_axes[0:3, i],
                np.cross(new_joint_poses_home[0:3, i], new_joint_axes[0:3, i])))
        new_thetas = (new_thetas @
            tm([0, 0, 0, 0, np.pi, 0]) @ tm([0, 0, 0, 0, 0, np.pi]))
        if np.size(self.link_dimensions) != 1:
            new_link_dimensions = np.zeros((self.link_dimensions.shape))
            for i in range(self.link_dimensions.shape[1]):
                new_link_dimensions[0:3, i] = (
                    self.link_dimensions[0:3,(self.link_dimensions.shape[1] - i -1)])
            self.link_dimensions = new_link_dimensions
        if len(self.link_home_positions) != 1:
            new_link_home_positions = [None] * len(self.link_home_positions)
            for i in range(len(new_link_home_positions)):
                new_link_home_positions[i] = (
                    self.link_home_positions[len(new_link_home_positions) - i -1])
            self.link_home_positions = new_link_home_positions
        self.screw_list = new_screw_list
        self.original_screw_list = np.copy(new_screw_list)
        #print(self.base_pos_global, "")
        new_end_effector_home = new_thetas @ self.end_effector_home_local
        self.base_pos_global = new_thetas
        self.original_joint_poses_home = new_joint_poses_home
        self.joint_poses_home = np.zeros((3, new_screw_list.shape[1]))
        self.screw_list_body = np.zeros((6, new_screw_list.shape[1]))
        if new_joint_poses_home.size > 1:
            for i in range(0, new_screw_list.shape[1]):
                self.joint_poses_home[0:3, i] = fsr.TrVec(new_thetas, new_joint_poses_home[0:3, i])
                #Convert TrVec
        for i in range(0, new_screw_list.shape[1]):
            self.screw_list[:, i] = fmr.Adjoint(new_thetas.gTM()) @ new_screw_list[:, i]
            if new_joint_poses_home.size <= 1:
                [w, th, joint_pose_temp, h] = fmr.TwistToScrew(self.screw_list[:, i])
                #Convert TwistToScrew
                self.joint_poses_home[0:3, i] = joint_pose_temp; # For plotting purposes
        self.end_effector_home = new_end_effector_home
        self.original_end_effector_home = self.end_effector_home.copy()
        if len(self.link_home_positions) != 1:
            new_link_mass_transforms = [None] * len(self.link_home_positions)
            new_link_mass_transforms[0] = self.link_home_positions[0];
            for i in range(1, 6):
                new_link_mass_transforms[i] = (
                    self.link_home_positions[i-1].inv() @ self.link_home_positions[i])
            new_link_mass_transforms[len(self.link_home_positions) -1] = (
                self.link_home_positions[5].inv() @ self.end_effector_home)
            self.link_mass_transforms = new_link_mass_transforms
        self.box_spatial_links = 0
        for i in range(0, new_screw_list.shape[1]):
            self.screw_list_body[:, i] = (
                fmr.Adjoint(self.end_effector_home.inv().gTM()) @ self.screw_list[:, i])
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

    def lineTrajectory(self, target, initial = 0, execute = True,
        tol = np.array([.05, .05, .05, .05, .05, .05]), delt = .01):
        """
        Move the arm end effector in a straight line towards the target
        Args:
            target: Target pose to reach
            intial: Starting pose. If set to 0, as is default, uses current position
            execute: Execute the desired motion after calculation
            tol: tolerances on motion
            delt: delta in meters to be calculated for each step
        Returns:
            theta_list list of theta configurations
        """
        if initial == 0:
            initial = self.end_effector_pos_global.copy()
        satisfied = False
        init_theta = np.copy(self._theta)
        theta_list = []
        count = 0
        while not satisfied and count < 2500:
            count+=1
            error = fsr.Error(target, initial).gTAA().flatten()
            satisfied = True
            for i in range(6):
                if abs(error[i]) > tol[i]:
                    satisfied = False
            initial = fsr.CloseGap(initial, target, delt)
            theta_list.append(np.copy(self._theta))
            self.IK(initial, self._theta)
        self.IK(target, self._theta)
        theta_list.append(self._theta)
        if (execute == False):
            self.FK(init_theta)
        return theta_list


    def visualServoToTarget(self, target, tol = 2, ax = 0, plt = 0, fig = 0):
        """
        Use a virtual camera to perform visual servoing to target
        Args:
            target: Object to move to
            tol: piexel tolerance
            ax: matplotlib object to draw to
            plt: matplotlib plot
            fig: whether or not to draw
            Returns: Thetalist for arm, figure object
        Returns:
            theta: thetas at goal
            fig: figure
        """
        if (len(self.cameras) == 0):
            print("NO CAMERA CONNECTED")
            return
        at_target = False
        done = False
        start_pos = self.FK(self._theta)
        theta = 0
        j = 0
        plt.ion()
        images = []
        while not (at_target and done):
            for i in range(len(self.cameras)):
                pose_adjust = tm()
                at_target = True
                done = True
                img, q, suc = self.cameras[i][0].getPhoto(target)
                if not suc:
                    print("Failed to locate Target")
                    return self._theta
                if img[0] < self.cameras[i][2][0] - tol:
                    pose_adjust[0] = -.01
                    at_target = False
                if img[0] > self.cameras[i][2][0] + tol:
                    pose_adjust[0] = .01
                    at_target = False
                if img[1] < self.cameras[i][2][1] - tol:
                    pose_adjust[1] = -.01
                    at_target = False
                if img[1] > self.cameras[i][2][1] + tol:
                    pose_adjust[1] = .01
                    at_target = False
                if at_target:
                    d = fsr.Distance(self.end_effector_pos_global, target)
                    print(d)
                    if d < .985:
                        done = False
                        pose_adjust[2] = -.01
                    if d > 1.015:
                        done = False
                        pose_adjust[2] = .01
                start_pos =start_pos @ pose_adjust
                theta = self.IK(start_pos, self._theta)
                self.updateCams()
                if fig != 0:
                    ax = plt.axes(projection = '3d')
                    ax.set_xlim3d(-7, 7)
                    ax.set_ylim3d(-7, 7)
                    ax.set_zlim3d(0, 8)
                    DrawArm(self, ax)
                    DrawRectangle(target, [.2, .2, .2], ax)
                    print("Animating")
                    plt.show()
                    plt.savefig('VideoTemp' + "/file%03d.png" % j)
                    ax.clear()
            j = j + 1
        return theta, fig

    def PDControlToGoalEE(self, theta, goal_position, Kp, Kd, prevtheta, max_theta_dot):
        """
        Uses PD Control to Maneuver to an end effector goal
        Args:
            theta: start theta
            goal_position: goal position
            Kp: P parameter
            Kd: D parameter
            prevtheta: prev_theta parameter
            max_theta_dot: maximum joint velocities
        Returns:
            scaled_theta_dot: scaled velocities
        """
        current_end_effector_pos = self.FK(theta)
        previous_end_effector_pos = self.FK(prevtheta)
        error_ee_to_goal = fsr.Norm(current_end_effector_pos[0:3, 3]-goal_position[0:3, 3])
        delt_distance_to_goal = (error_ee_to_goal-
            fsr.Norm(previous_end_effector_pos[0:3, 3]-goal_position[0:3, 3]))
        scale = Kp @ error_ee_to_goal + Kd @ min(0, delt_distance_to_goal)

        twist = self.TwistSpaceToGoalEE(theta, goal_position)
        twist_norm = fsr.TwistNorm(twist)
        normalized_twist = twist/twist_norm
        theta_dot = self.ThetadotSpace(theta, normalized_twist)
        scaled_theta_dot = max_theta_dot/max(abs(theta_dot)) @ theta_dot @ scale
        return scaled_theta_dot



    """
       _____      _   _                                     _    _____      _   _
      / ____|    | | | |                    /\             | |  / ____|    | | | |
     | |  __  ___| |_| |_ ___ _ __ ___     /  \   _ __   __| | | (___   ___| |_| |_ ___ _ __ ___
     | | |_ |/ _ \ __| __/ _ \ '__/ __|   / /\ \ | '_ \ / _` |  \___ \ / _ \ __| __/ _ \ '__/ __|
     | |__| |  __/ |_| ||  __/ |  \__ \  / ____ \| | | | (_| |  ____) |  __/ |_| ||  __/ |  \__ \
      \_____|\___|\__|\__\___|_|  |___/ /_/    \_\_| |_|\__,_| |_____/ \___|\__|\__\___|_|  |___/

    """

    def setDynamicsProperties(self, link_mass_transforms = None,
        link_home_positions = None, box_spatial_links = None, link_dimensions = None):
        """
        Set dynamics properties of the arm
        At mimimum dimensions are a required parameter for drawing of the arm.
        Args:
            link_mass_transforms: The mass matrices of links
            link_home_positions: List of Home Positions
            box_spatial_links: Mass Matrices (Inertia)
            link_dimensions: Dimensions of links
        """
        self.link_mass_transforms = link_mass_transforms
        self.link_home_positions =  link_home_positions
        self.box_spatial_links = box_spatial_links
        self.link_dimensions = link_dimensions

    def setMasses(self, mass):
        """
        set Masses
        Args:
            mass: mass
        """
        self.masses = mass

    def testArmValues(self):
        """
        prints a bunch of arm values
        """
        np.set_printoptions(precision=4)
        np.set_printoptions(suppress=True)
        print("S")
        print(self.screw_list, title = "screw_list")
        print("screw_list_body")
        print(self.screw_list_body, title = "screw_list_body")
        print("Q")
        print(self.joint_poses_home, title = "joint_poses_home list")
        print("end_effector_home")
        print(self.end_effector_home, title = "end_effector_home")
        print("original_end_effector_home")
        print(self.original_end_effector_home, title = "original_end_effector_home")
        print("_Mlinks")
        print(self.link_mass_transforms, title = "Link Masses")
        print("_Mhome")
        print(self.link_home_positions, title = "_Mhome")
        print("_Glinks")
        print(self.box_spatial_links, title = "_Glinks")
        print("_dimensions")
        print(self.link_dimensions, title = "Dimensions")


    def getJointTransforms(self):
        """
        returns tmobjects for each link in a serial arm
        Returns:
            tmlist
        """
        dimensions = np.copy(self.link_dimensions).conj().T
        joint_pose_list = [None] * dimensions.shape[0]

        end_effector_home = self.base_pos_global
        end_effector_transform = tm(fmr.FKinSpace(end_effector_home.gTM(),
            self.screw_list[0:6, 0:0], self._theta[0:0]))
        #print(end_effector_transform, "EEPOS")
        joint_pose_list[0] = end_effector_transform
        for i in range((self.screw_list.shape[1])):
            if self.link_home_positions == None:
                temp_tm = tm()
                temp_tm[0:3, 0] = self.original_joint_poses_home[0:3, i]
                end_effector_home = self.base_pos_global @ temp_tm
            else:
                end_effector_home = self.link_home_positions[i]
            #print(end_effector_home, "end_effector_home" + str(i + 1))
            #print(self._theta[0:i+1])
            end_effector_transform = tm(fmr.FKinSpace(end_effector_home.gTM(),
                self.screw_list[0:6, 0:i], self._theta[0:i]))
            #print(end_effector_transform, "EEPOS")
            joint_pose_list[i] = end_effector_transform
        if dimensions.shape[0] > self.screw_list.shape[1]:
            #Fix handling of dims
            #print(fsr.TAAtoTM(np.array([0.0, 0.0, self.link_dimensions[-1, 2], 0.0 , 0.0, 0.0])))
            joint_pose_list[len(joint_pose_list) - 1] = self.FK(self._theta)
        return joint_pose_list

    def setArbitraryHome(self, theta,T):
        """
        #  Given a pose and some T in the space frame, find out where
        #  that T is in the EE frame, then find the home pose for
        #  that arbitrary pose
        Args:
            theta: theta configuration
            T: new transform
        """

        end_effector_temp = self.FK(theta)
        ee_to_new = np.cross(np.inv(end_effector_temp),T)
        self.end_effector_home = np.cross(self.end_effector_home, ee_to_new)

    #Converted to Python - Joshua
    def restoreOriginalEE(self):
        """
        Retstore the original End effector of the Arm
        """
        self.end_effector_home = self.original_end_effector_home

    def getEEPos(self):
        """
        Gets End Effector Position
        """
        return self.end_effector_pos_global.copy()



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

    def staticForces(self, theta, end_effector_wrench):
        """
        Calculate forces on each link of the serial arm
        Args:
            theta: Current position of the arm
            end_effector_wrench: end effector wrench (space frame)
        Returns:
            forces in newtons on each joint
        """
        end_effector_temp = self.FK(theta) #Space Frame
        tau = self.jacobian(theta).conj().T @ end_effector_wrench
        return tau

    #def staticForces(self, theta, end_effector_wrench):
    #    end_effector_temp = self.FK(theta)
    #    wrenchS = fmr.Adjoint(ling.inv(end_effector_temp)).conj().T @ end_effector_wrench
    #    return self.jacobian(theta).conj().T @ wrenchS

    def staticForcesInv(self, theta, tau):
        """
        Given a position on the arm and forces for each joint,
        calculate the wrench on the end effector
        Args:
            theta: current joint positions of the arm
            tau: forces on the joints of the arm in Newtons
        Returns:
            wrench on the end effector of the arm
        """
        x_init = np.zeros((len(theta)))
        temp = lambda x : (self.staticForces(theta, x[0:6])-tau)
        end_effector_wrench = sci.optimize.fsolve(temp, x_init)
        return end_effector_wrench[0:6]

    def inverseDynamics(self, theta, theta_dot, theta_dot_dot, grav, end_effector_wrench):
        """
        Inverse dynamics
        Args:
            theta: theta
            theta_dot: theta 1st deriviative
            theta_dot_dot: theta 2nd derivative
            grav: gravity
            end_effector_wrench: end effector wrench
        Returns
            tau: tau
            A: todo
            V: todo
            vel_dot: todo
            F: todo
        """
        return self.inverseDynamicsE(theta, theta_dot, theta_dot_dot, grav, end_effector_wrench)

    def inverseDynamicsEMR(self, theta, theta_dot, theta_dot_dot, grav, end_effector_wrench):
        """
        Inverse dynamics
        Args:
            theta: theta
            theta_dot: theta 1st deriviative
            theta_dot_dot: theta 2nd derivative
            grav: gravity
            end_effector_wrench: end effector wrench
        Returns
            tau: tau
            A: todo
            V: todo
            vel_dot: todo
            F: todo
        """
        return fmr.inverseDynamics(theta, theta, theta_dot, grav, end_effector_wrench,
            self.link_mass_transforms, self.box_spatial_links, self.screw_list)

    def inverseDynamicsE(self, theta, theta_dot, theta_dot_dot, grav, end_effector_wrench):
        """
        Inverse dynamics
        Args:
            theta: theta
            theta_dot: theta 1st deriviative
            theta_dot_dot: theta 2nd derivative
            grav: gravity
            end_effector_wrench: end effector wrench
        Returns
            tau: tau
            A: todo
            V: todo
            vel_dot: todo
            F: todo
        """
        #Multiple Bugs Fixed - Liam Aug 4 2019
        A = np.zeros((self.screw_list.shape))
        V = np.zeros((self.screw_list.shape))
        vel_dot = np.zeros((self.screw_list.shape))

        for i in range(self.screw_list.shape[1]):
            #A[0:6, i] =(fmr.Adjoint(ling.inv(self.link_home_positions[i,:,:].reshape((4, 4)))) @
            #   self.screw_list[0:6, i].reshape((6, 1))).reshape((6))
            A[0:6, i] = (self.link_home_positions[i].inv().Adjoint() @
                self.screw_list[0:6, i]).reshape((6))

            #Ti_im1 =
            #   (fmr.MatrixExp6(fmr.VecTose3(A[0:6, i]) * theta[i]) @
            #   ling.inv(self.link_mass_transforms[i,:,:])
            Ti_im1 = (fmr.MatrixExp6(fmr.VecTose3(A[0:6, i]) * theta[i]) @
                self.link_mass_transforms[i].inv().TM)
            if i > 0:
                V[0:6, i] = (A[0:6, i].reshape((6, 1)) * theta_dot[i] +
                    fmr.Adjoint(Ti_im1) @ V[0:6, i-1].reshape((6, 1))).reshape((6))
                #print((((A[0:6, i] * theta_dot_dot[i]).reshape((6, 1)) + (fmr.Adjoint(Ti_im1) @
                #   vel_dot[0:6, i-1]).reshape((6, 1)) + (fmr.ad(V[0:6, i]) @ A[0:6, i] *
                #   theta_dot[i]).reshape((6, 1))).reshape((6, 1)), "vcomp"))
                vel_dot[0:6, i] = (((A[0:6, i] * theta_dot_dot[i]).reshape((6, 1)) +
                    (fmr.Adjoint(Ti_im1) @ vel_dot[0:6, i-1]).reshape((6, 1)) +
                    (fmr.ad(V[0:6, i]) @ A[0:6, i] * theta_dot[i]).reshape((6, 1))).reshape((6)))
            else:
                V[0:6, i] = ((A[0:6, i].reshape((6, 1)) * theta_dot[i] +
                    fmr.Adjoint(Ti_im1) @ np.zeros((6, 1))).reshape((6)))
                vel_dot[0:6, i] = (((A[0:6, i] * theta_dot_dot[i]).reshape((6, 1)) +
                    (fmr.Adjoint(Ti_im1) @ np.vstack((np.array([[0],[0],[0]]),
                    grav))).reshape((6, 1)) +
                    (fmr.ad(V[0:6, i]) @ A[0:6, i] * theta_dot[i]).reshape((6, 1))).reshape((6)))
        F = np.zeros((self.screw_list.shape))
        tau = np.zeros((theta.size, 1))
        for i in range(self.screw_list.shape[1]-1, -1, -1):
            if i == self.screw_list.shape[1]-1:
                #continue
                Tip1_i = self.link_mass_transforms[i+1].inv().TM
                F[0:6, i] = (fmr.Adjoint(Tip1_i).conj().T @ end_effector_wrench +
                    self.box_spatial_links[i,:,:] @ vel_dot[0:6, i] - fmr.ad(V[0:6, i]).conj().T @
                    self.box_spatial_links[i,:,:] @ V[0:6, i])
            else:
                #print(( fmr.MatrixExp6(-fmr.VecTose3((A[0:6, i+1].reshape((6, 1))) *
                #   theta(i + 1))) @ ling.inv(self.link_mass_transforms[i+1,:,:]), "problem"))
                Tip1_i = (fmr.MatrixExp6(-fmr.VecTose3(A[0:6, i+1]) * theta[i + 1]) @
                    self.link_mass_transforms[i+1].inv().TM)
                F[0:6, i] = (fmr.Adjoint(Tip1_i).conj().T @ F[0:6, i+1] +
                    self.box_spatial_links[i,:,:] @ vel_dot[0:6, i] -
                    fmr.ad(V[0:6, i]).conj().T @ self.box_spatial_links[i,:,:] @ V[0:6, i])

            tau[i] = F[0:6, i].conj().T @ A[0:6, i]
        return tau, A, V, vel_dot, F

    def inverseDynamicsC(self, theta, theta_dot, theta_dot_dot, grav, end_effector_wrench):
        """
        Inverse dynamics Implementation of algorithm in Lynch 8.4
        Args:
            theta: theta
            theta_dot: theta 1st deriviative
            theta_dot_dot: theta 2nd derivative
            grav: gravity
            end_effector_wrench: end effector wrench
        Returns
            tau: tau
            M: todo
            c: todo
            G: todo
            ee: todo
        """
        n = theta.size
        A = np.zeros((6*n, n))
        G = np.zeros((6*n, n))
        for i in range (n):
            A[(i-1)*6+1:(i-1)*6+6, i] = (
                fmr.Adjoint(ling.inv(self.link_home_positions[i,:,:])) @ self.screw_list[0:6, i])
            G[(i-1)*6+1:(i-1)*6+6,(i-1)*6+1:(i-1)*6+7] = self.box_spatial_links[i,:,:]
        joint_axes = np.zeros((6*n, 6*n))
        Vbase = np.zeros((6*n, 1))
        T10 = ling.inv(self.FKLink(theta, 1))
        vel_dot_base = (
            np.hstack((self.Adjoint(T10) @ np.array([[0],[0],[0],[-grav]]), np.zeros((5*n, 1)))))
        Ttipend = ling.inv(self.FK(theta)) @ self.FKLink(theta, n)
        Ftip = np.vstack((np.zeros((5*n, 1)), fmr.Adjoint(Ttipend).conj().T @ end_effector_wrench))
        for i in range (1, n):
            Ti_im1 = ling.inv(self.FKlink(theta, i)) @ self.FKLink(theta, i-1)
            joint_axes[(i-1) * 6 + 1:(i-1) *6 + 6, (i-2)*6+1:(i-2)*6+6] = fmr.Adjoint(Ti_im1)
        L = ling.inv(np.identity((6*n))-joint_axes)
        V = L @ (A @ theta_dot + Vbase)
        adV = np.zeros((6*n, 6*n))
        adAthd = np.zeros((6*n, 6*n))
        for i in range(1, n):
            adV[(i-1) * 6 + 1:(i-1) * 6+6,(i-1)*6+1:(i-1)*6+6] = fmr.ad(V[(i-1)*6+1:(i-1)*6+6, 0])
            adAthd[(i-1)*6+1:(i-1) * 6 + 6, (i - 1) * 6 + 1 : (i - 1) * 6 + 6] = (
                fmr.ad(theta_dot[i] @ A[(i - 1) * 6 + 1 : (i - 1)* 6 + 6, i]))
        vel_dot = L @ (A @ theta_dot_dot - adAthd @ joint_axes @ V - adAthd @ Vbase @vel_dot_base)
        F = L.conj().T @ (G @ vel_dot - adV.conj().T @ G @ V + Ftip)
        tau = A.conj().T @ F
        M = A.conj().T @ L.conj().T @ G @ L @ A

        return tau, M, G

    def forwardDynamicsE(self, theta, theta_dot, tau, grav, end_effector_wrench):
        """
        Forward dynamics
        Args:
            theta: theta
            theta_dot: theta 1st deriviative
            tau:joint torques
            grav: gravity
            end_effector_wrench: end effector wrench
        Returns
            theta_dot_dot: todo
            M: todo
            h: todo
            ee: todo
        """
        M = self.massMatrix(theta)
        h = self.coriolisGravity(theta, theta_dot, grav)
        ee = self.endEffectorForces(theta, end_effector_wrench)
        theta_dot_dot = ling.inv(M) @ (tau-h-ee)

        return theta_dot_dot, M, h, ee

    def forwardDynamics(self, theta, theta_dot, tau, grav, end_effector_wrench):
        """
        Forward dynamics
        Args:
            theta: theta
            theta_dot: theta 1st deriviative
            tau:joint torques
            grav: gravity
            end_effector_wrench: end effector wrench
        Returns
            theta_dot_dot: todo
        """
        theta_dot_dot = fmr.forwardDynamics(theta, theta_dot, tau, grav,
            end_effector_wrench, self.link_mass_transforms, self.box_spatial_links, self.screw_list)
        return theta_dot_dot

    def massMatrix(self, theta):
        """
        calculates mass matrix for configuration
        Args:
            theta: theta for configuration
        Returns:
            M: mass matrix
        """
        #Debugged - Liam 8/4/19
        M = np.zeros(theta.size)
        for i in range(theta.size):
            Ji = self.jacobianLink(theta, i)
            jt = Ji.conj().T @ self.box_spatial_links[i,:,:] @ Ji
            #M = M + jt
        #print(M, "M1")
        #print(fmr.massMatrix(theta, self.link_mass_transforms,
        #    self.box_spatial_links, self.screw_list), "Masses")
        return M

    def coriolisGravity(self, theta, theta_dot, grav):
        """
        Implements Coriolis Gravity from dynamics
        Args:
            theta: theta config
            theta_dot: theta deriv
            grav: gravity
        Returns:
            coriolisGravity
        """
        h = self.inverseDynamicsE(theta, theta_dot, 0*theta, grav, np.zeros((6, 1)))
        return h

    def endEffectorForces(self, theta, end_effector_wrench):
        """
        Calculates forces at the end effector
        Args:
            theta: joint configuration
            end_effector_wrench: wrench at the end effector
        Returns:
            forces at the end effector
        """
        grav = np.array([[0.0],[0.0],[-9.81]])
        return self.inverseDynamicsE(theta, 0*theta, 0*theta, np.zeros((3, 1)), end_effector_wrench)



    """
           _                 _     _                _____      _            _       _   _
          | |               | |   (_)              / ____|    | |          | |     | | (_)
          | | __ _  ___ ___ | |__  _  __ _ _ __   | |     __ _| | ___ _   _| | __ _| |_ _  ___  _ __  ___
      _   | |/ _` |/ __/ _ \| '_ \| |/ _` | '_ \  | |    / _` | |/ __| | | | |/ _` | __| |/ _ \| '_ \/ __|
     | |__| | (_| | (_| (_) | |_) | | (_| | | | | | |___| (_| | | (__| |_| | | (_| | |_| | (_) | | | \__ \
      \____/ \__,_|\___\___/|_.__/|_|\__,_|_| |_|  \_____\__,_|_|\___|\__,_|_|\__,_|\__|_|\___/|_| |_|___/

    """

    #Converted to Python - Joshua
    def jacobian(self, theta):
        """
        Calculates Space Jacobian for given configuration
        Args:
            theta: joint configuration
        Returns:
            jacobian
        """
        return fmr.JacobianSpace(self.screw_list, theta)

    #Converted to Python - Joshua
    def jacobianBody(self, theta):
        """
        Calculates Body Jacobian for given configuration
        Args:
            theta: joint configuration
        Returns:
            jacobian
        """
        return fmr.JacobianBody(self.screw_list_body, theta)

    #Converted to Python - Joshua
    #Fixed Bugs - Liam
    def jacobianLink(self, theta, i):
        """
        Calculates Space Jacobian for given configuration link
        Args:
            theta: joint configuration
            i: joint index
        Returns:
            jacobian
        """
        t_ad = self.FKLink(theta, i).inv().Adjoint()
        t_js = fmr.JacobianSpace(self.screw_list[0:6, 0:i], theta[0:i])
        t_z = np.zeros((6, len(theta) - 1))
        t_mt = t_ad @ t_js
        return np.hstack((t_mt, t_z))

    def jacobianEE(self, theta):
        """
        Calculates End Effector Jacobian for given configuration
        Args:
            theta: joint configuration
        Returns:
            jacobian
        """
        jacobian = self.jacobian(theta)
        return (self.FK(theta).inv() @ jacobian).Adjoint()
        #return fmr.Adjoint()

    def jacobianEEtrans(self, theta):
        """
        Calculates Jacobian for given configuration
        Args:
            theta: joint configuration
        Returns:
            jacobian
        """
        end_effector_temp = self.FK(theta)
        end_effector_temp[0:3, 0:3] = np.identity((3))
        jacobian = self.jacobian(theta)
        return fmr.Adjoint(ling.inv(end_effector_temp)) @ jacobian

    def numericalJacobian(self, theta):
        """
        Calculates numerical Jacobian for given configuration
        Args:
            theta: joint configuration
        Returns:
            jacobian
        """
        jacobian = np.zeros((6, theta.size))
        temp = lambda x : np.reshape(self.FK(x),((1, 16)))
        numerical_jacobian = fsr.NumJac(temp, theta, 0.006)
        for i in range(0, np.size(theta)):
            jacobian[0:6, i] = (fmr.se3ToVec(ling.inv(self.FK(theta).conj().T) @
                np.reshape(numerical_jacobian[:, i],((4, 4))).conj().T))

        return jacobian

    def getManipulability(self, theta = None):
        """
        Calculates Manipulability at a given configuration
        Args:
            theta: configuration
        Returns:
            Manipulability parameters
        """
        if theta == None:
            theta = self._theta.copy()
        Jb = self.jacobianBody(theta)
        Jw = Jb[0:3,:] #Angular
        Jv = Jb[3:6,:] #Linear

        Aw = Jw @ Jw.T
        Av = Jv @ Jv.T

        AwEig, AwEigVec = np.linalg.eig(Aw)
        AvEig, AvEigVec = np.linalg.eig(Av)

        uAw = 1/(np.sqrt(max(AwEig))/np.sqrt(min(AwEig)))
        uAv = 1/(np.sqrt(max(AvEig))/np.sqrt(min(AvEig)))

        return AwEig, AwEigVec, uAw, AvEig, AvEigVec, uAv

    """
       _____
      / ____|
     | |     __ _ _ __ ___   ___ _ __ __ _
     | |    / _` | '_ ` _ \ / _ \ '__/ _` |
     | |___| (_| | | | | | |  __/ | | (_| |
      \_____\__,_|_| |_| |_|\___|_|  \__,_|

    """


    def addCamera(self, cam, end_effector_to_cam):
        """
        adds a camera to the arm
        Args:
            cam: camera object
            end_effector_to_cam: end effector to camera transform
        """
        cam.moveCamera(self.end_effector_pos_global @ end_effector_to_cam)
        img, joint_poses_home, suc = cam.getPhoto(self.end_effector_pos_global @
            tm([0, 0, 1, 0, 0, 0]))
        camL = [cam, end_effector_to_cam, img]
        self.cameras.append(camL)
        print(self.cameras)

    def updateCams(self):
        """
        Updates camera locations
        """
        for i in range(len(self.cameras)):
            self.cameras[i][0].moveCamera(self.end_effector_pos_global @ self.cameras[i][1])

    """
       _____ _                 __  __      _   _               _
      / ____| |               |  \/  |    | | | |             | |
     | |    | | __ _ ___ ___  | \  / | ___| |_| |__   ___   __| |___
     | |    | |/ _` / __/ __| | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
     | |____| | (_| \__ \__ \ | |  | |  __/ |_| | | | (_) | (_| \__ \
      \_____|_|\__,_|___/___/ |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    """

    def move(self, T, stationary = False):
        """
        Moves the arm to another location
        Args:
            T: new base location
            stationary: boolean for keeping the end effector in origianal location while
                moving the base separately
        """
        curpos = self.end_effector_pos_global.copy()
        curth = self._theta.copy()
        self.initialize(T, self.original_screw_list_body,
            self.end_effector_home_local, self.original_joint_poses_home)
        if stationary == False:
            self.FK(self._theta)
        else:
            self.IK(curpos, curth)

    def draw(self, ax):
        """
        Draws the arm using the faser_plot library
        """
        DrawArm(self, ax)
