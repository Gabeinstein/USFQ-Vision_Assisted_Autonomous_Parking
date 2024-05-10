#! /usr/bin/env python3

#Proyecto Sistemas de Comunicacion - Primavera 2024
#Integrantes: Gabriel Ona, Jose Montahuano y Emilia Casares

### Action Planner
### Mira el status de /parking/type y de /parking/enable 
### Realiza el movimiento adecuado de parqueo

#ROS LIBRARIES
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_msgs.msg import Bool

#DATA MANAGEMENT LIBRARIES
import scipy.io
import os

#Paths
script_dir = os.path.dirname(os.path.realpath(__file__))
main_dir = script_dir.replace('/scripts','',-1)

retro_velocity_path = os.path.join(main_dir,'path_planning/retro','velocity_profile.mat')
retro_angular_path = os.path.join(main_dir,'path_planning/retro','angular_profile.mat')

paralelo_velocity_path = os.path.join(main_dir,'path_planning/paralelo','velocity_profile.mat')
paralelo_angular_path = os.path.join(main_dir,'path_planning/paralelo','angular_profile.mat')

diagonal_velocity_path = os.path.join(main_dir,'path_planning/diagonal','velocity_profile.mat')
diagonal_angular_path = os.path.join(main_dir,'path_planning/diagonal','angular_profile.mat')

frente_velocity_path = os.path.join(main_dir,'path_planning/frente','velocity_profile.mat')
frente_angular_path = os.path.join(main_dir,'path_planning/frente','angular_profile.mat')

#Profiles
retro_velocity_profile = scipy.io.loadmat(retro_velocity_path).get('velocity_profile')
retro_angular_profile = scipy.io.loadmat(retro_angular_path).get('angular_profile')

paralelo_velocity_profile = scipy.io.loadmat(paralelo_velocity_path).get('velocity_profile')
paralelo_angular_profile = scipy.io.loadmat(paralelo_angular_path).get('angular_profile')

diagonal_velocity_profile = scipy.io.loadmat(diagonal_velocity_path).get('velocity_profile')
diagonal_angular_profile = scipy.io.loadmat(diagonal_angular_path).get('angular_profile')

frente_velocity_profile = scipy.io.loadmat(frente_velocity_path).get('velocity_profile')
frente_angular_profile = scipy.io.loadmat(frente_angular_path).get('angular_profile')

#Variables
current_velocity_profile = None
current_angular_profile = None
parking_enable = False
parking_type = ""
counter = 0
dt = 0.1
ant_vel = 0
step_time = 0

#Pub
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

def callback_parking_type(msg):
    global parking_type
    global dt
    global current_velocity_profile
    global current_angular_profile

    if (msg.data == "retro"):
        parking_type = "retro"
        current_velocity_profile = retro_velocity_profile
        current_angular_profile = retro_angular_profile
    elif (msg.data == "paralelo"):
        parking_type = "paralelo"
        current_velocity_profile = paralelo_velocity_profile
        current_angular_profile = paralelo_angular_profile
    elif (msg.data == "diagonal"):
        parking_type = "diagonal"
        current_velocity_profile = diagonal_velocity_profile
        current_angular_profile = diagonal_angular_profile
    elif (msg.data == "frente"):
        parking_type = "frente"
        current_velocity_profile = frente_velocity_profile
        current_angular_profile = frente_angular_profile

def callback_parking_enable(msg):
    global parking_enable
    if (msg.data == True):
        parking_enable = True


def update_velocities(next_vel_lin_x, next_vel_ang_z):
    global ant_vel
    global step_time
    msg = Twist()
    msg.linear.x = next_vel_lin_x
    msg.angular.z = next_vel_ang_z
    if (ant_vel != next_vel_lin_x):
        #print((rospy.Time.now() - step_time).to_sec())
        step_time = rospy.Time.now()
    ant_vel = next_vel_lin_x

    #rospy.loginfo("---------- Status ----------")
    #rospy.loginfo("Linear Velocity X: " + str(msg.linear.x))
    #rospy.loginfo("Angular Velocity Z: " + str(msg.angular.z))
    #rospy.loginfo("----------------------------")

    pub.publish(msg)

def print_charging_bar(counter, max_count):
    charging_chars = "▮"
    empty_chars = "▯"
    bar_length = 20
    charging_percentage = min(100, (counter / max_count) * 100)
    num_charging_chars = int((charging_percentage / 100) * bar_length)
    num_empty_chars = bar_length - num_charging_chars
    charging_bar = charging_chars * num_charging_chars + empty_chars * num_empty_chars
    print(f"\rALGORITHM PROGRESS -> [{charging_bar}] {charging_percentage:.2f}%", end="", flush=True)


def main():
    global parking_enable
    global parking_type
    global counter
    global dt
    global current_velocity_profile
    global current_angular_profile
    global step_time

    enable_topic = rospy.Subscriber('/parking/enable', Bool, callback_parking_enable)
    type_topic = rospy.Subscriber('/parking/type', String, callback_parking_type)
    pub_parking = rospy.Publisher('/parking/enable', Bool, queue_size=1)
    counter = 0
    start_time = rospy.Time.now()
    step_time = rospy.Time.now()
    while not rospy.is_shutdown():
        if (parking_enable):
            if (parking_type != ""):
                if (counter >= len(current_velocity_profile)):
                    msg = Twist()
                    msg.linear.x = 0
                    msg.angular.z = 0
                    pub.publish(msg)
                    parking_enable = False
                    parking_type = ""
                    counter = 0
                    #print("")
                    #print((rospy.Time.now() - start_time).to_sec())
                    msg = Bool()
                    msg.data = False   
                    pub_parking.publish(msg)
                else:
                    print_charging_bar(counter,len(current_velocity_profile)-1)
                    update_velocities(current_velocity_profile[counter],current_angular_profile[counter])

                if ((rospy.Time.now() - start_time).to_sec() > dt):
                    counter += 1
                    start_time = rospy.Time.now()
        
        rate.sleep() 
if __name__ == "__main__":
    rospy.init_node('path_planner_node')
    rate = rospy.Rate(1000)
    main()