#!/usr/bin/env python3
# encoding: utf-8

import rospy
from geometry_msgs.msg import Twist
import sys, select, termios, tty
from std_msgs.msg import Bool

msg = """
Control Your SLAM-Bot!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%
space key, k : force stop
anything else : stop smoothly
p : parking

CTRL-C to quit
"""

moveBindings = {
    'i': (1, 0),
    'o': (1, -1),
    'j': (0, 1),
    'l': (0, -1),
    'u': (1, 1),
    ',': (-1, 0),
    '.': (-1, 1),
    'm': (-1, -1),
    'I': (1, 0),
    'O': (1, -1),
    'J': (0, 1),
    'L': (0, -1),
    'U': (1, 1),
    'M': (-1, -1),
}

speedBindings = {
    'Q': (1.1, 1.1),
    'Z': (.9, .9),
    'W': (1.1, 1),
    'X': (.9, 1),
    'E': (1, 1.1),
    'C': (1, .9),
    'q': (1.1, 1.1),
    'z': (.9, .9),
    'w': (1.1, 1),
    'x': (.9, 1),
    'e': (1, 1.1),
    'c': (1, .9),
}

parkingBindings = {
    'p': (1),
    'P': (1),
}


def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist: key = sys.stdin.read(1)
    else: key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def vels(speed, turn):
    return "currently:\tspeed %s\tturn %s " % (speed, turn)


if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin)
    rospy.init_node('transbot_keyboard')
    linear_limit = rospy.get_param('~linear_limit', 0.45)
    angular_limit = rospy.get_param('~angular_limit', 2.0)
    pub = rospy.Publisher('~/cmd_vel', Twist, queue_size=1)
    pub_parking_enable = rospy.Publisher('/parking/enable',Bool,queue_size=1)
    (speed, turn) = (0.2, 1.0)
    (x, th) = (0, 0)
    status = 0
    count = 0
    try:
        print(msg)
        print(vels(speed, turn))
        while (1):
            key = getKey()
            if key in moveBindings.keys():
                x = moveBindings[key][0]
                th = moveBindings[key][1]
                count = 0
            elif key in speedBindings.keys():
                speed = speed * speedBindings[key][0]
                turn = turn * speedBindings[key][1]
                count = 0
                if speed > linear_limit: speed = linear_limit
                if turn > angular_limit: turn = angular_limit
                print(vels(speed, turn))
                if (status == 14): print(msg)
                status = (status + 1) % 15
            elif key in parkingBindings.keys():
                enable_parking = Bool()
                enable_parking.data = True
                pub_parking_enable.publish(enable_parking)

            elif key == ' ': 
                (x, th) = (0, 0)
                enable_parking = Bool()
                enable_parking.data = False
                pub_parking_enable.publish(enable_parking)
            else:
                count = count + 1
                if count > 4: (x, th) = (0, 0)
                if (key == '\x03'): break
  
            twist = Twist()
            twist.linear.x = speed * x
            twist.angular.z = turn * th
            pub.publish(twist)
    except Exception as e: print(e)
    finally: pub.publish(Twist())
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
