# vision_assisted_autonomous_parking

Class project of Digital Systems Communications Spring 2024

- PARKING APPLICATION
  -- Proyecto Sistemas de Comunicaciones Primavera 2024 --
  Integrantes: Gabriel Ona, Jose Montahuano y Emilia Casares

---

## Descripcion

Este proyecto consiste en un sistema de deteccion de entornos para la aplicacion de parqueo autonomo.
El robot que se utilizo fue (Transbot - Yahboom) con ROS Melodic y una computadora NVidia Jetson Nano con CUDA 10.2

Se creo un entorno en donde se encuentran 4 diferentes tipos de parqueo.
Paralelo
Retro
Diagonal
Frente

El sistema por medio de una red neuronal convolucional de arquitectura AlexNET, determina el entorno de parqueo en el que se
encuentra mediante el sensor LiDAR incorporado. Este sensor entrega un vector con 720 rangos entre 0.5 a 12. Los 720 datos pasan por preprocesamiento
y K Best Feature Selection antes de entrar a la input layer de la red neuronal. Esta red dispone de 4 clases de clasificacion por lo que podra determinar
cuando el robot se encuentra en un escenario sobre el cual puede realizar la maniobra de parqueo.

La arquitectura del sistema de nodos se detalla a continuacion: - Nodos:
usfq_keyboard -> Publisher: /cmd_vel /parking/enable Subscriber: /parking/enable
usfq_vision_ai -> Publisher: /parking/type Subscriber: /scan
usfq_path_planning -> Publisher: /cmd_vel /parking/enable Subscriber: /parking/enable /parking/type - Topicos:
/cmd_vel
/scan
/parking/enable
/parking/type

---

## Funcionamiento

Control Remoto:
q w e  
 a s d
z x c

p : parking

1/2 : incrementar/decrementar velocidad max por 10%
3/4 : incrementar/decrementar solo velocidad lineal por 10%
5/6 : incrementar/decrementar solo velocidad angular por 10%
space key, k : force stop

## Cuando se presiona la tecla de "P" parking, el sistema de control remoto deja de estar abilitado y el robot realiza la accion de parqueo.

PUEDE COMENZAR A UTILIZAR EL SISTEMA USFQ VISION ASSISTED AUTONOMOUS PARKING
"Maneje con precaucion"

---
