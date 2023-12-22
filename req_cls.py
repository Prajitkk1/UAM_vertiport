# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 12:51:19 2022

@author: praji
"""
import random
import math 
import numpy as np
import time
from sympy.geometry import Segment3D

class ports:
    """
    A class which has details about all the port locations.
    """
    def __init__(self,drone_count):
        self.normal_ports = [[0,0, -2], [-3,0,-2]]
        self.battery_ports = [[-2,3,-2]]
        # self.fake_ports = [[0,3,-5], [-11,1,-5], [-8,6,-5], [-8,-5,-5], [-12,8,-5]] #These are closer
        # self.hover_spots = [[-1,-9,-1.25], [-9,-9,-1.25],  [-8,10,-1.25], [-2,4,-1.25]]
        self.fake_ports = [[-8.5, 5, -5], [-9, -4, -5], [-4, 10, -5], [-3, -7, -5], [2, 8, -5], [4, -6.5, -5]]
        self.hover_spots =[[-5, 5.5, -1.25], [-5, -3, -1.25], [-6, 1.5, -1.25], [-1, 6.5, -1.25], [-0.5, -4, -1.25], [2, 2.5, -1.25]]

        self.no_ports = len(self.normal_ports)
        self.no_battery_ports = len(self.battery_ports)
        self.no_hoverspots = len(self.hover_spots)
        self.port_status = {}
        self.port_center_loc =[0,0,-4] #Filler
        self.drone_count = drone_count
        self.dist_threshold = 10
        self.reset_ports()

    def reset_ports(self):
        for i in range(self.no_ports):
            self.port_status[i] = {"port_no": i, "position":self.normal_ports[i],"occupied": False, "type":0}
            
        self.battery_port_status = {}
        for i in range(self.no_battery_ports):
            self.battery_port_status[i] = {"port_no": i,"position":self.battery_ports[i],"occupied": False, "type": 1}
        
        self.hover_spot_status = {}
        for i in range(self.no_hoverspots):
            self.hover_spot_status[i] = {"port_no": i,"position":self.hover_spots[i],"occupied": False, "type": 2}

        self.no_total = self.no_ports + self.no_battery_ports + self.no_hoverspots
        self.feature_mat = np.zeros((self.no_total, 4)) #four features per port
            
    def update_port(self,port):
        if port:
            if port['type'] == 'normal':
                # print('\nport relinquished\n')
                self.change_status_normal_port(port['port_no'],False)
            elif port['type'] == 'battery':
                # print('\nbattery relinquished\n')
                self.change_status_battery_port(port['port_no'],False)
            elif port['type'] == 'hover':
                # print('\nhover port relinquished\n')
                self.change_hover_spot_status(port['port_no'],False)

    def update_all(self):
        '''''
        This function will iterate through all ports, battery ports, and hover spots and 
        update the vertiport feature matrix accordingly
        '''''
        for i in range(self.no_ports):
            if self.port_status[i]['occupied'] == True:
                availability = 0
            else:
                availability = 1
            node_type = 0
            self.feature_mat[i] = [availability, node_type, self.port_status[i]["position"][0], self.port_status[i]["position"][1]]
        for i in range(self.no_battery_ports):
            if self.battery_port_status[i]['occupied'] == True:
                availability = 0
            else:
                availability = 1
            node_type = 1
            self.feature_mat[i+self.no_ports] = [availability, node_type, self.battery_port_status[i]["position"][0], self.battery_port_status[i]["position"][1]]
        for i in range(self.no_hoverspots):
            if self.hover_spot_status[i]['occupied'] == True:
                availability = 0
            else:
                availability = 1
            node_type = 2
            self.feature_mat[i+self.no_ports+self.no_battery_ports] = [availability, node_type, self.hover_spot_status[i]["position"][0], self.hover_spot_status[i]["position"][1]]
        # print(self.feature_mat[:,0])

    def get_all_empty_ports(self):
        return {"normal_ports":self.port_status, "battery_ports":self.battery_port_status, "hover_spots": self.hover_spot_status}
            
            
    def get_empty_port(self):
        for i in range(self.no_ports):
            if self.port_status[i]["occupied"] == False:
                self.change_status_normal_port(self.port_status[i]['port_no'],True)
                return self.port_status[i]
    
    def get_empty_battery_port(self):
        for i in range(self.no_battery_ports):
            if self.battery_port_status[i]["occupied"] == False:
                self.change_status_battery_port(self.battery_port_status[i]['port_no'],True)
                return self.battery_port_status[i]
        return None
    
    def get_empty_hover_status(self):
        for i in range(self.no_hoverspots):
            if self.hover_spot_status[i]["occupied"] == False:
                self.change_hover_spot_status(self.hover_spot_status[i]['port_no'],True)
                return self.hover_spot_status[i]

    def get_destination(self, choice = 0, number = None):
        if number:
            return self.fake_ports[number]
        if choice == 0:
            return random.choice(self.fake_ports)
        else:
            empty_port = self.get_empty_hover_status()
            return empty_port['position']

    def change_status_normal_port(self, port_no, occupied):
        self.port_status[port_no]["occupied"] = occupied
        
    def change_status_battery_port(self, port_no, occupied):
        self.battery_port_status[port_no]["occupied"] = occupied
    
    def change_hover_spot_status(self, port_no, occupied):
        self.hover_spot_status[port_no]["occupied"] = occupied
            
    def get_count_empty_port(self):
        cnt = 0
        for i in range(self.no_ports):
            if self.port_status[i]["occupied"] == False:
                cnt+=1
        return cnt
    
    def get_count_empty_battery_port(self):
        cnt = 0
        for i in range(self.no_battery_ports):
            if self.port_status[i]["occupied"] == False:
                cnt+=1
        return cnt
    
    def get_count_empty_hover_Spot(self):
        cnt = 0
        for i in range(self.no_hoverspots):
            if self.hover_spot_status[i]["occupied"] == False:
                cnt+=1

        return cnt   
    
    def get_availability_ports(self,drone_locs):
        empty_ports = self.get_count_empty_port()
        uams_inside = self.count_uavs_inside(drone_locs)
        percent = empty_ports/uams_inside
        if uams_inside > 0:
            percent = empty_ports/uams_inside
            if percent > 0.8:
                return 2
            elif percent> 0.5:
                return 1
            else:
                return 0
        else:
            return 2


    def get_availability_battery_ports(self,drone_locs):
        empty_ports = self.get_count_empty_battery_port()
        uams_inside = self.count_uavs_inside(drone_locs)
        if uams_inside > 0:
            percent = empty_ports/uams_inside
            if percent > 0.8:
                return 2
            elif percent> 0.5:
                return 1
            else:
                return 0
        else:
            return 2
        
    def get_availability_hover_spots(self,drone_locs):
        empty_ports = self.get_count_empty_hover_Spot()
        uams_inside = self.count_uavs_inside(drone_locs)
        percent = empty_ports/uams_inside
        if uams_inside > 0:
            percent = empty_ports/uams_inside
            if percent > 0.8:
                return 2
            elif percent> 0.5:
                return 1
            else:
                return 0
        else:
            return 2
    
    def get_port_status(self, port, port_type): #Changed from port_status to avoid key errors
        if port_type == 'normal':
                 return self.port_status[port]["occupied"]
        elif port_type == 'battery':
                return self.battery_port_status[port]["occupied"]
        elif port_type == 'hover':
                return self.hover_spot_status[port]["occupied"]
    
    def count_uavs_inside(self,drone_locs):
        UAVs_inside = 0
        for i in range(len(drone_locs)):
            dist= self._calculate_distance(drone_locs[i])
            if dist<self.dist_threshold: #Switched from > to <
                UAVs_inside +=1
        return UAVs_inside
    
    def _calculate_distance(self,cur_location):

        return np.linalg.norm(np.array(self.port_center_loc)-np.array(cur_location)) #math.dist starts at python3.8, I'm using 3.7 lol
    
    def get_all_port_statuses(self):
        
        return [self.port_status , self.battery_port_status, self.hover_spot_status]
    


class UAMs:
    def __init__(self, drone_name,offset):
        self.drone_name = drone_name
        self.drone_no = drone_name # use split and get the drone number alone
        self.velocity = 1 # 1 m/s
        self.all_battery_states = {'critical':0,'sufficient':1,'full':2} #Added 4.25.22 -> 3 different battery states to go with the overall battery remaining
        self.battery_state = 2
        self.battery_remaining = 100
        self.distance_travelled = 0
        self.next_takeoff = None
        self.next_landing = None
        self.all_states = {"in-air":0, "in-port":1, "battery-port":2, "in-action":3, "in-destination":4}
        self.job_status = {"initial_loc":None, "final_dest":None, "current_pos": None}
        self.status = 1
        self.status_to_set = 1
        self.offset = offset
        self.current_location = []
        self.previous_location = []
        self.in_portzone = False
        self.port_center_loc =[0,0,-4] #Filler
        self.dist_threshold = 10
        self.drone_locs = [[0,0,-1],[6,0,-1],[-2,3,-1],[6,4,-1],[-3,0,-1], [-3,0,1]]
        self.current_location = None
        self.in_battery_port = 0
        self.port_identification = None
        self.upcoming_schedule = {"landing-time": 0, "takeoff-time":0, 'delay':None, 'total-delay':0, 'time':0, 'end-port':None}
        self.env_time = 0
        self.schedule_status = 0 
        self.tasks_completed = 0
        self.good_takeoffs = 0
        self.clock_speed = 200
        self.sleep_time = 0.5 / self.clock_speed

    def get_status(self):
        if self.status == self.all_states['in-air']:
            status = 0
        else:
            status = 1
        return status
    
    def set_status(self,status, final_status):
        self.status = self.all_states[status]
        self.status_to_set = self.all_states[final_status]
    
    def get_schedule_state(self):

        if (self.upcoming_schedule['landing-time'] - 1) <= self.upcoming_schedule['time'] <= (self.upcoming_schedule['landing-time'] + 1):
            schedule = 0
        else:
            schedule = 1
        if (self.upcoming_schedule['takeoff-time'] - 1) <= self.upcoming_schedule['time'] <= (self.upcoming_schedule['takeoff-time'] + 1):
            schedule = 0
        else:
            schedule = 1
        return schedule 
    
    def set_schedule(self,whatever):
        pass
    
    def get_battery_state(self):
        return self.battery_state 
    
    def calculate_reduction(self,old_position,new_position): 
        time_travelled = np.linalg.norm(np.array(old_position)-np.array(new_position)) / self.velocity
        discharge_rate = 0.50

        if time_travelled < 1 and self.status == self.all_states["in-air"]:
            return 2
        if time_travelled < 1 and self.status in [self.all_states["in-port"], self.all_states["battery-port"]]:
            return 4

        return discharge_rate * time_travelled

    def update_battery(self, reduce):
        self.battery_remaining -= reduce
        if self.battery_remaining < 0:
            self.battery_remaining = 0
        if self.battery_remaining == 100:
            self.battery_state = self.all_battery_states['full']
        elif 30 <= self.battery_remaining <= 100: #Added 4.25.22 
            self.battery_state = self.all_battery_states['sufficient'] #Ditto
        elif 0 <= self.battery_remaining <= 30:
            self.battery_state = self.all_battery_states['critical'] #Ditto
        
        
    def distance_to_nearest_drone(self, drone_no):
        #we can use it later
        pass


    
    def check_zone(self):
        dist = self._calculate_distance(self.current_location)
        if dist<self.dist_threshold:
            self.in_portzone = True
        else:
            self.in_portzone = False
    
    def update(self, current_loc, client,port,env_time):
        self.upcoming_schedule["time"] = env_time
        self.env_time = env_time
        # print(self.env_time)
        reduction = self.calculate_reduction(self.previous_location, self.current_location)
        self.update_battery(reduction)
        self.previous_location = self.current_location

        if self.status == self.all_states['in-action']: 
            if self.status_to_set == self.all_states['in-destination']: 
                dist = self._calculate_distance(current_loc[:-1],self.job_status['final_dest'][:-1])
                if dist < 0.75: #Drone reached destination and is ready for the next task
                    self.set_status('in-destination','in-action')
                    self.tasks_completed += 1
                else:
                    final_pos = self.job_status['final_dest']
                    client.moveToPositionAsync(final_pos[0],final_pos[1],final_pos[2], velocity=1, vehicle_name=self.drone_name)

            elif self.status_to_set == self.all_states['battery-port']:
                dist = self._calculate_distance(current_loc[:-1],self.job_status['final_dest'][:-1])
                if dist < 0.75: #Drone reached the battery port and is ready to charge
                    client.landAsync(vehicle_name = self.drone_name)
                    self.in_battery_port = 1
                    self.set_status('battery-port','in-action')
                    self.battery_remaining += 10
                else:
                    final_pos = self.job_status['final_dest']
                    client.moveToPositionAsync(final_pos[0],final_pos[1],final_pos[2], velocity=1, vehicle_name=self.drone_name)
                    time.sleep(self.sleep_time)

            elif self.status_to_set == self.all_states['in-air']:
                dist = self._calculate_distance(current_loc[:-1],self.job_status['final_dest'][:-1])
                if dist < 0.75: #Drone reached the hover spot and is ready for the next task
                    client.hoverAsync(vehicle_name = self.drone_name)
                    old_position = current_loc
                    new_position = self.job_status['final_dest']
                    # reduce = self.calculate_reduction(old_position,new_position) #Ditto
                    # self.update_battery(reduce) #Ditto
                    self.set_status('in-air','in-action')
                else:
                    final_pos = self.job_status['final_dest']
                    client.moveToPositionAsync(final_pos[0],final_pos[1],final_pos[2], velocity=1, vehicle_name=self.drone_name)
                    time.sleep(self.sleep_time)
            elif self.status_to_set == self.all_states['in-port']:
                dist = self._calculate_distance(current_loc[:-1],self.job_status['final_dest'][:-1])
                if dist < 1: #Drone reached destination and is ready for the next task
                    self.set_status('in-port','in-action')
                else:
                    final_pos = self.job_status['final_dest']
                    client.moveToPositionAsync(final_pos[0],final_pos[1],final_pos[2], velocity=1, vehicle_name=self.drone_name)
                    time.sleep(self.sleep_time)

        elif self.status == self.all_states['battery-port']:
            if self.battery_remaining >= 100:
                self.battery_remaining = 100
                self.set_status('battery-port','in-action')
            else:
                self.battery_remaining += 10

        elif self.status == self.all_states['in-port']:

            self.set_status('in-port','in-action')

        elif self.status == self.all_states['in-air']:
            pass

        elif self.status == self.all_states['in-destination']:
            self.assign_schedule(port,client,choice=1) #Assigning a hover port
            self.port_identification = {'type':'hover','port_no':port.hover_spots.index(self.job_status['final_dest'])}
            des = self.job_status['final_dest']
            final_pos = self.get_final_pos(des, self.offset)
            client.moveToPositionAsync(final_pos[0],final_pos[1],final_pos[2], velocity=1, vehicle_name=self.drone_name)
            time.sleep(self.sleep_time)
            self.set_status('in-action','in-air')
        
    def _calculate_distance(self,cur_location, dest):
        return np.linalg.norm(np.array(dest)-np.array(cur_location))
    
    def update_port(self, is_it):
        self.in_battery_port = is_it
        
        
    def get_final_pos(self,port, offset):
        # print([port , offset, [port[0] +offset[0] , port[1] + offset[1], port[2]]])
        return [port[0] - offset[0] , port[1] - offset[1], port[2]]
    
    def assign_schedule(self,port,client,choice = 0):
        """
        Instead of the schedule class, everytime the drone reaches its destination(fake ports) I assign another schedule. 

        Returns
        -------
        None.

        """
        self.job_status['final_dest'] = port.get_destination(choice)
        random_landing = random.randint(1,3) * self.clock_speed 
        random_takeoff = random.randint(2,4) * self.clock_speed
        self.upcoming_schedule["landing-time"] = random_landing + self.env_time
        self.upcoming_schedule["takeoff-time"] = random_takeoff + self.env_time
        if self.upcoming_schedule["delay"]:
            self.upcoming_schedule["total-delay"] += self.upcoming_schedule["delay"]
        self.upcoming_schedule["delay"] = None
        self.upcoming_schedule["end-port"] = self.job_status['final_dest']


    def get_state_status(self):
        """
        Our state space(On-time to takeoff/land (0,1)) indicates the takeoff and landing time, delay. Please calculate them here
        For perfect takeoff - you can have threshold of 1 minute. Create new variable, check its timing if it is good timing set it to 1 else 0
        if there is delay just calculate them for the reward claculation
        1. For landing the delay time is from the time mentioned in the self.upcoming_schedule["Landing-time"]
        2. for takeoff the delay time is from the time mentioned in the self.upcoming_schedule["takeoff-time"]

        Returns
        -------
        None.

        """
        threshold = 300 # 5 minutes
        if self.status == self.all_states['in-air'] or self.status == self.all_states['in-action']:
            if (self.upcoming_schedule["landing-time"] - threshold <= self.upcoming_schedule['time'] <= self.upcoming_schedule["landing-time"] + threshold):
                self.schedule_status = 0
                return 0
            elif ( self.upcoming_schedule['time'] <= self.upcoming_schedule["landing-time"] - threshold):
                self.schedule_status = 1
                return 1
            elif (self.upcoming_schedule["landing-time"] + threshold <= self.upcoming_schedule['time']): # Late
                self.schedule_status = 2
                self.upcoming_schedule["delay"] = self.upcoming_schedule["time"] - self.upcoming_schedule["landing-time"] 
                return 2
        elif self.status == self.all_states['in-port'] or self.status == self.all_states['battery-port']:
            if (self.upcoming_schedule["takeoff-time"] - threshold <= self.upcoming_schedule['time'] <= self.upcoming_schedule["takeoff-time"] + threshold):
                self.schedule_status = 0
                return 0
            elif (self.upcoming_schedule['time'] <= self.upcoming_schedule["takeoff-time"] - threshold):
                self.schedule_status = 1
                return 1
            elif (self.upcoming_schedule["takeoff-time"] + threshold <= self.upcoming_schedule['time']):
                self.schedule_status = 2
                self.upcoming_schedule["delay"] = self.upcoming_schedule["time"] - self.upcoming_schedule["takeoff-time"]
                return 2
        else:
            self.schedule_status = 0
            return 0
        
        
    def get_all_status(self):
        state = {"location": self.current_location, "state": self.status, "schedule": self.upcoming_schedule, "battery": self.battery_remaining}
        return state

        
    


