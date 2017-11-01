from klampt import *
from common import *
import random
import numpy as np
import math

##################### SETTINGS ########################
event = 'A'

#difficulty
#difficulty = 'easy'
difficulty = 'medium'
#difficulty = 'hard'

omniscient_sensor = True

#random_seed = 12345
random_seed = random.seed()

verbose = False

################ STATE ESTIMATION #####################

class MyObjectStateEstimator:
    """Your own state estimator that will provide a state estimate given
    CameraColorDetectorOutput readings."""
    def __init__(self):
        self.reset()
        #TODO: fill this in with your own camera model, if you wish
        self.Tsensor = None
        cameraRot = [0,-1,0,0,0,-1,1,0,0]
        if event == 'A':
            #at goal post, pointing a bit up and to the left
            self.Tsensor = (so3.mul(so3.rotation([0,0,1],0.20),so3.mul(so3.rotation([0,-1,0],0.25),cameraRot)),[-2.55,-1.1,0.25])
        elif event == 'B':
            #on ground near robot, pointing up and slightly to the left
            self.Tsensor = (so3.mul(so3.rotation([1,0,0],-0.10),so3.mul(so3.rotation([0,-1,0],math.radians(90)),cameraRot)),[-1.5,-0.5,0.25])
        else:
            #on ground near robot, pointing to the right
            self.Tsensor = (cameraRot,[-1.5,-0.5,0.25])
        self.fov = 90
        self.w,self.h = 320,240
        self.dmax = 5
        self.dt = 0.02
        self.focalLength = math.tan(math.radians(self.fov*0.5))*self.w/2
        self.blobDict = {}
        return
    def reset(self):
        pass
    def update(self,observation):
        """Produces an updated MultiObjectStateEstimate given a CameraColorDetectorOutput
        sensor reading."""
        obj = []
        for blob in observation.blobs:
            dist = (0.2*self.focalLength)/blob.w
            hyp = math.sqrt(2)

            if blob.x < self.w/2.0:
                oppx = blob.x * (1.0/(self.w/2.0))
                anglex = math.asin(oppx/hyp) + math.radians(45)
            else:
                rightxpixel = self.w - blob.x
                oppx = rightxpixel * (1.0/(self.w/2.0))
                anglex = math.asin(oppx/hyp) + math.radians(45)

            if blob.y < self.h/2.0:
                oppy = blob.y * (1.0/(self.h/2.0))
                angley = math.asin(oppy/hyp) + math.radians(45)
            else:
                rightypixel = self.h - blob.y
                oppy = rightypixel * (1.0/(self.h/2.0))
                angley = math.asin(oppy/hyp) + math.radians(45)

            #z = dist * math.cos(anglex) * math.cos(angley)
            z = dist
            scale = self.focalLength/z
            x = (blob.x - self.w/2.0) / scale
            y = (blob.y - self.h/2.0) / scale

            worldCoor = se3.apply(self.Tsensor,(x,y,z))

            if worldCoor[2] < -0.107:
                if blob.color in self.blobDict:
                   del self.blobDict[blob.color]
                continue

            if blob.color not in self.blobDict:
                self.blobDict[blob.color] = []
            self.blobDict[blob.color].append(worldCoor)

            if len(self.blobDict[blob.color]) < 2:
                return MultiObjectStateEstimate([])

            xvelocity = (self.blobDict[blob.color][-1][0] - self.blobDict[blob.color][-2][0])/(self.dt)
            yvelocity = (self.blobDict[blob.color][-1][1] - self.blobDict[blob.color][-2][1])/(self.dt)
            zvelocity = (self.blobDict[blob.color][-1][2] - self.blobDict[blob.color][-2][2])/(self.dt)
            posVel = [worldCoor[0],worldCoor[1],worldCoor[2],xvelocity,yvelocity,zvelocity]

            obj.append(ObjectStateEstimate(blob.color,posVel))
        #print self.blobDict
        return MultiObjectStateEstimate(obj)

################### CONTROLLER ########################

class MyController:
    """Attributes:
    - world: the WorldModel instance used for planning.
    - objectStateEstimator: a StateEstimator instance, which you may set up.
    - state: a string indicating the state of the state machine. TODO:
      decide what states you want in your state machine and how you want
      them to be named.
    """
    def __init__(self,world,robotController):
        self.world = world
        self.objectStateEstimator = None
        self.state = None
        self.robotController = robotController
        self.reset(robotController)
        self.robot = robotController.model()
        self.links = [self.robot.getLink(i) for i in range(0,self.robot.numLinks()-1)]
        self.ballIndex = 0
        self.t = 0
        self.colors = []

    def reset(self,robotController):
        """Called on initialization, and when the simulator is reset.
        TODO: You may wish to fill this in with custom initialization code.
        """
        self.objectStateEstimator = MyObjectStateEstimator()
        self.objectEstimates = None
        self.state = 'initialize'
        #TODO: you may want to do more here to set up your
        #state machine and other initial settings of your controller.
        #The 'waiting' state is just a placeholder and you are free to
        #change it as you see fit.
        #self.qdes = [0.0, -3.12413936106985, -0.5672320068981571, 1.5655603390389137, 1.0000736613927508, -0.32637657012293964, 0.0]
        self.qdes = [0.0, 2.0961404316451895, -0.312413936106985, 1.7418385934903409, 1.0000736613927508, -0.32637657012293964, 0.0]
        self.initVis()
        pass

    def IKsolve(self,point,linkIndex):
        obj = ik.objective(self.links[linkIndex],local=(0.0,0.142,0.0285),world=(point[0],point[1],point[2]))
        #obj2 = ik.fixed_objective(self.links[2], local=(0,0,0))
        ik.solve_global(obj,iters = 1000,tol = 1e-3,numRestarts = 50)
            #print "IK Failure! Link: " + str(linkIndex)
            #self.IKsolve(point,linkIndex-1)
        self.qdes = self.robot.getConfig()
        return

    def finalPosition(self, position, velocity, index):
        gravity = 9.8
        t = (position[0]+1.70)/abs(velocity[0])
        finalPos = vectorops.sub(vectorops.madd(position,velocity,t),[0,0,0.5*gravity*t*t])
        if finalPos[2] < 0.102 and index > 0:
            peakHeightTime = -velocity[2]/-gravity
            peakHeight = vectorops.sub(vectorops.madd(position,velocity,peakHeightTime),[0,0,0.5*gravity*peakHeightTime*peakHeightTime])[2]
            if peakHeight > 0.108:
                t = math.sqrt((2*(0.102-peakHeight))/(-gravity)) + peakHeightTime
                finalPos = vectorops.sub(vectorops.madd(position,velocity,t),[0,0,0.5*gravity*t*t])
                groundVelocity = -gravity*(t-peakHeightTime)
                return self.finalPosition(finalPos, (0.7*velocity[0], 0.7*velocity[1], -0.6*groundVelocity), index-1)
        return finalPos

    def myPlayerLogic(self,
                      dt,
                      sensorReadings,
                      objectStateEstimate,
                      robotController):
        """
        TODO: fill this out to updates the robot's low level controller
        in response to a new time step.  This is allowed to set any
        attributes of MyController that you wish, such as self.state.

        Arguments:
        - dt: the simulation time elapsed since the last call
        - sensorReadings: the sensor readings given on the current time step.
          this will be a dictionary mapping sensor names to sensor data.
          The name "blobdetector" indicates a sensor reading coming from the
          blob detector.  The name "omniscient" indicates a sensor reading
          coming from the omniscient object sensor.  You will not need to
          use raw sensor data directly, if you have a working state estimator.
        - objectStateEstimate: a MultiObjectStateEstimate class (see
          stateestimation.py) produced by the state estimator.
        - robotController: a SimRobotController instance giving access
          to the robot's low-level controller.  You can call any of the
          methods.  At the end of this call, you can either compute some
          PID command via robotController.setPIDCommand(), or compute a
          trajectory to execute via robotController.set/addMilestone().
          (if you are into masochism you can use robotController.setTorque())
        """
        qcmd = robotController.getCommandedConfig()
        vcmd = robotController.getCommandedVelocity()
        qsns = robotController.getSensedConfig()
        vsns = robotController.getSensedVelocity()
        meanPositions = [objectStateEstimate.get(o.name).meanPosition() for o in objectStateEstimate.objects]
        meanVelocities = [objectStateEstimate.get(p.name).meanVelocity() for p in objectStateEstimate.objects]
        names = [o.name for o in objectStateEstimate.objects]
        #print meanVelocities
        #setting a PID command can be accomplished with the following
        #robotController.setPIDCommand(self.qdes,[0.0]*7)

        #queuing up linear interpolation can be accomplished with the following
        #dt = 0.5   #how much time it takes for the robot to reach the target
        #robotController.appendLinear(self.qdes,dt)

        #queuing up a fast-as possible interpolation can be accomplished with the following
        #robotController.addMilestone(self.qdes)

        self.t += dt
        if self.state == 'moving' or self.state == 'initialize':
            self.k = self.ballIndex
            if len(meanPositions) <= 4 and len(meanPositions) != 0 and list(sensorReadings)[0] == "blobdetector" and names[-1] not in set(self.colors):
                self.k = -1
                self.state = 'switch'
            elif self.ballIndex <= len(meanPositions)-1 and list(sensorReadings)[0] == "omniscient":
                if meanPositions[self.ballIndex][2] > 0.103 and self.t>1.0:
                    self.ballIndex += 1
                    self.state = 'switch'
                else:
                    for i in range(self.ballIndex, len(meanPositions)-1):
                        if meanPositions[i][2] > 0.15:
                            self.ballIndex = i
                            self.state = 'switch'

        elif self.state == 'switch':
            if len(meanPositions) > 0:
                endPosition = self.finalPosition(meanPositions[self.k], meanVelocities[self.k], 2)
            else:
                return
            if endPosition is not None:
                print endPosition
                self.IKsolve((endPosition[0], endPosition[1], endPosition[2]), 5)
                if len(names) > 0 and len(meanPositions) > 0:
                    self.colors.append(names[0])
                self.state = 'moving'

        robotController.setMilestone(self.qdes)
        #robotController.setLinear(self.qdes, dt)
        return

    def loop(self,dt,robotController,sensorReadings):
        """Called every control loop (every dt seconds).
        Input:
        - dt: the simulation time elapsed since the last call
        - robotController: a SimRobotController instance. Use this to get
          sensor data, like the commanded and sensed configurations.
        - sensorReadings: a dictionary mapping sensor names to sensor data.
          The name "blobdetector" indicates a sensor reading coming from the
          blob detector.  The name "omniscient" indicates a sensor reading coming
          from the omniscient object sensor.
        Output: None.  However, you should produce a command sent to
          robotController, e.g., robotController.setPIDCommand(qdesired).

        """
        multiObjectStateEstimate = None
        if self.objectStateEstimator and 'blobdetector' in sensorReadings:
            multiObjectStateEstimate = self.objectStateEstimator.update(sensorReadings['blobdetector'])
            self.objectEstimates = multiObjectStateEstimate
            #multiObjectStateEstimate is now a MultiObjectStateEstimate (see common.py)
        if 'omniscient' in sensorReadings:
            omniscientObjectState = OmniscientStateEstimator().update(sensorReadings['omniscient'])
            #omniscientObjectStateEstimate is now a MultiObjectStateEstimate (see common.py)
            multiObjectStateEstimate  = omniscientObjectState
            #if you want to visualize the traces, you can uncomment this
            #self.objectEstimates = multiObjectStateEstimate

        self.myPlayerLogic(dt,
                           sensorReadings,multiObjectStateEstimate,
                           robotController)

        self.updateVis()
        return

    def initVis(self):
        """If you want to do some visualization, initialize it here.
            TODO: You may consider visually debugging some of your code here, along with updateVis().
        """
        pass

    def updateVis(self):
        """This gets called every control loop.
        TODO: You may consider visually debugging some of your code here, along with initVis().

        For example, to draw a ghost robot at a given configuration q, you can call:
          kviz.add_ghost()  (in initVis)
          kviz.set_ghost(q) (in updateVis)

        The current code draws gravity-inflenced arcs leading from all the
        object position / velocity estimates from your state estimator.  Event C
        folks should set gravity=0 in the following code.
        """
        if self.objectEstimates:
            for o in self.objectEstimates.objects:
                #draw a point
                kviz.update_sphere("object_est"+str(o.name),o.x[0],o.x[1],o.x[2],0.03)
                kviz.set_color("object_est"+str(o.name),(o.name[0],o.name[1],o.name[2],1))
                #draw an arc
                trace = []
                x = [o.x[0],o.x[1],o.x[2]]
                v = [o.x[3],o.x[4],o.x[5]]
                if event=='C': gravity = 0
                else: gravity = 9.8
                for i in range(20):
                    t = i*0.05
                    trace.append(vectorops.sub(vectorops.madd(x,v,t),[0,0,0.5*gravity*t*t]))
                kviz.update_polyline("object_trace"+str(o.name),trace);
                kviz.set_color("object_trace"+str(o.name),(o.name[0],o.name[1],o.name[2],1))
