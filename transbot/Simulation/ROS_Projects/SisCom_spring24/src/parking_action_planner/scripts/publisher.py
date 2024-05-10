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
retro_time_vector_path = os.path.join(main_dir,'path_planning/retro', 'time_vector.mat')

paralelo_velocity_path = os.path.join(main_dir,'path_planning/paralelo','velocity_profile.mat')
paralelo_angular_path = os.path.join(main_dir,'path_planning/paralelo','angular_profile.mat')
paralelo_time_vector_path = os.path.join(main_dir,'path_planning/paralelo', 'time_vector.mat')

diagonal_velocity_path = os.path.join(main_dir,'path_planning/diagonal','velocity_profile.mat')
diagonal_angular_path = os.path.join(main_dir,'path_planning/diagonal','angular_profile.mat')
diagonal_time_vector_path = os.path.join(main_dir,'path_planning/diagonal', 'time_vector.mat')

frente_velocity_path = os.path.join(main_dir,'path_planning/frente','velocity_profile.mat')
frente_angular_path = os.path.join(main_dir,'path_planning/frente','angular_profile.mat')
frente_time_vector_path = os.path.join(main_dir,'path_planning/frente', 'time_vector.mat')

#Profiles
retro_velocity_profile = scipy.io.loadmat(retro_velocity_path).get('velocity_profile')
retro_angular_profile = scipy.io.loadmat(retro_angular_path).get('angular_profile')
retro_time_vector_profile = scipy.io.loadmat(retro_time_vector_path).get('time_vector')

paralelo_velocity_profile = scipy.io.loadmat(paralelo_velocity_path).get('velocity_profile')
paralelo_angular_profile = scipy.io.loadmat(paralelo_angular_path).get('angular_profile')
paralelo_time_vector_profile = scipy.io.loadmat(paralelo_time_vector_path).get('time_vector')

diagonal_velocity_profile = scipy.io.loadmat(diagonal_velocity_path).get('velocity_profile')
diagonal_angular_profile = scipy.io.loadmat(diagonal_angular_path).get('angular_profile')
diagonal_time_vector_profile = scipy.io.loadmat(diagonal_time_vector_path).get('time_vector')

frente_velocity_profile = scipy.io.loadmat(frente_velocity_path).get('velocity_profile')
frente_angular_profile = scipy.io.loadmat(frente_angular_path).get('angular_profile')
frente_time_vector_profile = scipy.io.loadmat(frente_time_vector_path).get('time_vector')

#Diferenciales de tiempo
dt_retro = retro_time_vector_profile[1] - retro_time_vector_profile[0]
dt_paralelo = paralelo_time_vector_profile[1] - paralelo_time_vector_profile[0]
dt_diagonal = diagonal_time_vector_profile[1] - diagonal_time_vector_profile[0]
dt_frente = frente_time_vector_profile[1] - frente_time_vector_profile[0]

#Variables
current_velocity_profile = None
current_angular_profile = None
parking_enable = False
parking_type = ""
counter = 0
dt = 0

#Pub
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

def callback_parking_type(msg):
    global parking_type
    global dt_diagonal
    global dt_frente
    global dt_paralelo
    global dt_retro
    global dt
    global current_velocity_profile
    global current_angular_profile

    if (msg.data == "retro"):
        parking_type = "retro"
        dt = dt_retro
        current_velocity_profile = retro_velocity_profile
        current_angular_profile = retro_angular_profile
    elif (msg.data == "paralelo"):
        parking_type = "paralelo"
        dt = dt_paralelo
        current_velocity_profile = paralelo_velocity_profile
        current_angular_profile = paralelo_angular_profile
    elif (msg.data == "diagonal"):
        parking_type = "diagonal"
        dt = dt_diagonal
        current_velocity_profile = diagonal_velocity_profile
        current_angular_profile = diagonal_angular_profile
    elif (msg.data == "frente"):
        parking_type = "frente"
        dt = dt_frente
        current_velocity_profile = frente_velocity_profile
        current_angular_profile = frente_angular_profile

def callback_parking_enable(msg):
    global parking_enable
    if (msg.data == True):
        parking_enable = True


def update_velocities(next_vel_lin_x, next_vel_ang_z):
    msg = Twist()
    msg.linear.x = next_vel_lin_x
    msg.angular.z = next_vel_ang_z

    rospy.loginfo("---------- Status ----------")
    rospy.loginfo("Linear Velocity X: " + str(msg.linear.x))
    rospy.loginfo("Angular Velocity Z: " + str(msg.angular.z))
    rospy.loginfo("----------------------------")

    pub.publish(msg)


def main():
    global parking_enable
    global parking_type
    global counter
    global dt
    global current_velocity_profile
    global current_angular_profile

    enable_topic = rospy.Subscriber('/parking/enable', Bool, callback_parking_enable)
    type_topic = rospy.Subscriber('/parking/type', String, callback_parking_type)

    counter = 0
    start_time = rospy.Time.now()

    while not rospy.is_shutdown():
        if (parking_enable):
            rospy.loginfo("( " + str(parking_enable) + ", " + str(type(parking_type)) + ", " + str(dt))
            if (parking_type != ""):
                if (counter >= len(current_velocity_profile)):
                    msg = Twist()
                    msg.linear.x = 0
                    msg.angular.z = 0
                    pub.publish(msg)
                    parking_enable = False
                    parking_type = ""
                    counter = 0
                    start_time = rospy.Time.now()
                else:
                    update_velocities(current_velocity_profile[counter],current_angular_profile[counter])
                
                if ((rospy.Time.now() - start_time).to_sec() > dt):
                    counter += 1
                    start_time = rospy.Time.now()
        rate.sleep()

if __name__ == "__main__":
    rospy.init_node('path_planner_node')
    rate = rospy.Rate(1000)
    main()