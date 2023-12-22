# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:38:39 2022

@author: Prajit
"""

from numpy import empty


class ActionManager:
    """
    This class should decode the actions and send the cooridnate to the Main environment
    """
    def __init__(self, port):
        self.port = port

    
    def action_decode(self,drone, action):
        """
        First check the status then assign the port/hoveringspot/takeoff/land .... etc

        Parameters
        ----------
        drone : drone object
            DESCRIPTION.
        action : just a scalar value [0-4]
            DESCRIPTION.

        Returns
        -------
        None.

        """
        status = drone.status
        if status == drone.all_states['in-air']:
            #here we are in landing phase, return 0 - 4 stay still, land, battery port, empty hovering spot
            if action ==0:
                return {"action": "stay", "position" : None}

            elif action ==1: #land
                empty_port = self.port.get_empty_port()
                if not empty_port:
                    return {"position" : None, "action": "stay"}
                else:
                    self.port.change_status_normal_port(empty_port["port_no"], True)
                    final_pos = self.get_final_pos(empty_port["position"], drone.offset)
                    drone.in_battery_port = 0
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'normal','port_no':empty_port["port_no"]}
                    # print(final_pos)
                    return {"position" : final_pos, "action": "land"}

            elif action ==2: #land in bat
                empty_port = self.port.get_empty_battery_port()
                if not empty_port:
                    return {"position" : None, "action": "stay"}
                else:
                    drone.in_battery_port = 1
                    final_pos = self.get_final_pos(empty_port["position"], drone.offset)
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'battery','port_no':empty_port["port_no"]}
                    return {"position" : final_pos, "action": "land-b"}
            else: #move to hover
                empty_port = self.port.get_empty_hover_status()
                final_pos = self.get_final_pos(empty_port["position"], drone.offset)
                self.port.update_port(drone.port_identification)
                drone.port_identification = {'type':'hover','port_no':empty_port["port_no"]}
                return {"position" : final_pos, "action": "hover"}
        elif status == drone.all_states['in-port'] or status == drone.all_states['battery-port']:
            #takeoff things
            if action ==0:
                return {"action": "stay"}
            elif action ==1: #takeoff
                dest = self.port.get_destination(choice=0)
                final_pos = self.get_final_pos(dest, drone.offset)
                self.port.update_port(drone.port_identification)
                return {"position" : final_pos, "action": "takeoff"}
                
            elif action ==2: #move to bat
                empty_port = self.port.get_empty_battery_port()    
                if not empty_port:
                    empty_port = self.port.get_empty_hover_status()
                    final_pos = self.get_final_pos(empty_port['position'],drone.offset)
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'hover','port_no':empty_port["port_no"]}
                    return {'position':final_pos, 'action':'takeoff-hover'}
                else:
                    final_pos = self.get_final_pos(empty_port["position"], drone.offset)
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'battery','port_no':empty_port["port_no"]}
                    return {"position" : final_pos, "action": "move-b"}
            else: #move from bat 
                empty_port = self.port.get_empty_port()
                if not empty_port:
                    empty_port = self.port.get_empty_hover_status()
                    final_pos = self.get_final_pos(empty_port['position'],drone.offset)
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'hover','port_no':empty_port["port_no"]}
                    return {'position':final_pos, 'action':'takeoff-hover'}
                else:
                    final_pos = self.get_final_pos(empty_port["position"], drone.offset)
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'normal','port_no':empty_port["port_no"]}
                    drone.in_battery_port = 0
                    return {"position" : final_pos, "action": "move"}
        return {'action':'continue'} # In the case that step is called while a robot is in-action
    
    def get_final_pos(self,port, offset):
        return [port[0] + offset[0] , port[1] + offset[1], port[2]]
        
    
 
    
 
    
class GL_ActionManager:
    """
    This class should decode the actions and send the cooridnate to the Main environment
    """
    def __init__(self, port):
        self.port = port
        self.actions = {0 : "stay still",
                        1 : "takeoff",
                        2 : "normal-1",
                        3 : "normal-2",
                        4 : "battery-1",
                        5 : "hover-1",
                        6 : "hover-2",
                        7 : "hover-3",
                        8 : "hover-4",
                        9 : "continue",
                        10 : "avoid collision"}

    
    def action_decode(self,drone, action):
        """
        action count:

            staystill - 0
            takeoff - 1
            move to normal port 1 - 2
            move to normal port 2 - 3
            move to battery port 1 - 4
            move to hover spot 1 - 5
            move to hover spot 2 - 6
            move to hover spot 3 - 7
            move to hover spot 4 - 8
            continue moving - 9
            avoid collision - 10
            

            
    
        First check the status then assign the port/hoveringspot/takeoff/land .... etc
        Update: We will use masking so no need to check the status
        Updateupdate: if using random walk might need to add extra conditionals to the if-elif statements

        Parameters
        ----------
        drone : drone object
            DESCRIPTION.
        action : just a scalar value [0-4]
            DESCRIPTION.


        Returns
        -------
        None.

        """
        action = self.actions[action[0]]   # Converting to a string for readability. 
        status = drone.status
        # print(f"action {action}")

        if action == "continue":
            return {'action':'continue'}

        if action == "stay still":
            return {"action" : "stay", "position" : None}
        
        elif action == "takeoff":
            # dest_num = int(action[-1]) - 1
            # dest = self.port.get_destination(choice=0)
            dest = drone.upcoming_schedule["end-port"]
            final_pos = self.get_final_pos(dest, drone.offset)
            self.port.update_port(drone.port_identification)

            return {"position" : final_pos, "action": "takeoff"}

        elif action in ["normal-1", "normal-2"]:
            norm_num = int(action[-1]) - 1
            if self.port.get_port_status(norm_num, 'normal') is False:                    
                final_pos = self.get_final_pos(self.port.port_status[norm_num]["position"], drone.offset)

                drone.in_battery_port = 0
                self.port.update_port(drone.port_identification)
                drone.port_identification = {'type':'normal','port_no':norm_num}

                #self.port.update_port(drone.port_identification)
                self.port.change_status_normal_port(norm_num, True)

                return {"position" : final_pos, "action": "land"}

        elif action == "battery-1":
            if self.port.get_port_status(0, 'battery') is False:
                final_pos = self.get_final_pos(self.port.battery_port_status[0]["position"], drone.offset)
                drone.in_battery_port = 1
                self.port.update_port(drone.port_identification)
                drone.port_identification = {'type':'battery', 'port_no':0}

                self.port.change_status_battery_port(0, True)
                #self.port.update_port(drone.port_identification)

                return {"position" : final_pos, "action": "land-b"}        
        
        elif action in ["hover-1", "hover-2", "hover-3", "hover-4"]:
            hover_num = int(action[-1]) - 1
            if not self.port.get_port_status(hover_num, "hover"):
                final_pos = self.get_final_pos(self.port.hover_spot_status[hover_num]["position"], drone.offset)
                drone.in_battery_port = 0
                self.port.update_port(drone.port_identification)
                drone.port_identification = {'type':'hover','port_no':hover_num}
                self.port.change_hover_spot_status(hover_num, True)
                if status in [drone.all_states["in-air"], drone.all_states["in-action"]]:
                    return {"position" : final_pos, "action": "hover"}
                else:
                    return {"position" : final_pos, "action": "takeoff-hover"}
        
        elif action == "avoid collision":
            return {"action" : "avoid collision", "position" : drone.job_status["final_dest"]}
    
        return {"action" : "continue"}  # Mainly for the random walk, since that can't use masking. 


        # if status == drone.all_states['in-air']:
        #     if action == "stay still":
        #         return {"action": "stay", "position" : None}
            
        #     elif action ==2:            #need to check if port is occupied before doing this action #todo
        #         #move/land in normal port - 1
        #         if self.port.get_port_status(0, 'normal') is False:                    
                    
        #             final_pos = self.get_final_pos(self.port.port_status[0]["position"], drone.offset)
        #             drone.in_battery_port = 0
        #             self.port.update_port(drone.port_identification)
        #             drone.port_identification = {'type':'normal','port_no':0}
        #             #self.port.update_port(drone.port_identification)
        #             self.port.change_status_normal_port(0, True)
        #             return {"position" : final_pos, "action": "land"}
        #         else:
        #             action_possible = False
        #     elif action ==3:            #need to check if port is occupied before doing this action #todo
        #         #move/land in normal port - 1
        #         if self.port.get_port_status(1, 'normal') is False:   
                    
        #             final_pos = self.get_final_pos(self.port.port_status[1]["position"], drone.offset)
        #             drone.in_battery_port = 0
        #             self.port.update_port(drone.port_identification)
        #             drone.port_identification = {'type':'normal','port_no':1}
        #             #self.port.update_port(drone.port_identification)
        #             self.port.change_status_normal_port(1, True)
        #             return {"position" : final_pos, "action": "land"}
        #         else:
        #             action_possible = False
            
        #     elif action ==4:            #need to check if port is occupied before doing this action #todo
        #         #move/land in normal port - 1
        #         if self.port.get_port_status(0, 'battery') is False:   
                    
        #             final_pos = self.get_final_pos(self.port.battery_port_status[0]["position"], drone.offset)
        #             drone.in_battery_port = 1
        #             self.port.update_port(drone.port_identification)
        #             drone.port_identification = {'type':'battery', 'port_no':0}
        #             self.port.change_status_battery_port(0, True)
        #             #self.port.update_port(drone.port_identification)
        #             return {"position" : final_pos, "action": "land-b"}        
        #         else:
        #             action_possible = False
                    
        #     elif action in [5, 6, 7, 8]: #Hovering actions
        #         hport = action - 5
        #         if not self.port.get_port_status(hport, "hover"):
        #             final_pos = self.get_final_pos(self.port.hover_spot_status[hport]["position"], drone.offset)
        #             drone.in_battery_port = 0
        #             self.port.update_port(drone.port_identification)
        #             drone.port_identification = {'type':'hover','port_no':hport}
        #             self.port.change_hover_spot_status(0, True)
        #             return {"position" : final_pos, "action": "hover"}
        #         else:
        #             action_possible = False
        #     else:
        #         print(f"action {action} not available")
        #         return {"action": "stay", "position" : None}
        # elif status == drone.all_states['in-port'] or status == drone.all_states['battery-port']:
        #     if action == 0:
        #         return {"action": "stay", "position" : None}


        #     elif action == 1:
        #         #takeoff
        #         dest = self.port.get_destination(choice=0)
        #         final_pos = self.get_final_pos(dest, drone.offset)
        #         self.port.update_port(drone.port_identification)
        #         return {"position" : final_pos, "action": "takeoff"}
            
            
        #     elif action ==2:            #need to check if port is occupied before doing this action #todo
        #         if self.port.get_port_status(0, 'normal') is False: 
        #             self.port.change_status_normal_port(0, True)
        #             final_pos = self.get_final_pos(self.port.port_status[0]["position"], drone.offset)
        #             drone.in_battery_port = 0
        #             self.port.update_port(drone.port_identification)
        #             drone.port_identification = {'type':'normal','port_no':0}
                    
        #             return {"position" : final_pos, "action": "move"}
            
        #     elif action ==3:            #need to check if port is occupied before doing this action #todo
        #         if self.port.get_port_status(1, 'normal') is False: 
        #             self.port.change_status_normal_port(1, True)
        #             final_pos = self.get_final_pos(self.port.port_status[1]["position"], drone.offset)
        #             drone.in_battery_port = 0
        #             self.port.update_port(drone.port_identification)
        #             drone.port_identification = {'type':'normal','port_no':1}
                    
        #             return {"position" : final_pos, "action": "move"}
        #         else:
        #             action_possible = False
        #     elif action ==4:            #need to check if port is occupied before doing this action #todo
        #         if self.port.get_port_status(0, 'battery') is False: 
        #             self.port.change_status_battery_port(0, True)
        #             final_pos = self.get_final_pos(self.port.battery_port_status[0]["position"], drone.offset)
        #             drone.in_battery_port = 1
        #             self.port.update_port(drone.port_identification)
        #             drone.port_identification = {'type':'battery', 'port_no':0}
        #             return {"position" : final_pos, "action": "move-b"}      
        #         else:
        #             action_possible = False
        #     ###########################################################################
        # else:
        #         return {"action": "stay", "position" : None}
        # if action_possible == False:
        #     return {'action':'continue'}
        # return {'action':'continue'} # In the case that step is called while a robot is in-action
    
    def get_final_pos(self,port, offset):
        return [port[0] + offset[0] , port[1] + offset[1], port[2]]
        
    
    
"""
All actions for reference
LAnding
1.1 stay still - 0
1.2 land in empty port - 1
1.3 land in battery port - 2
1.4 move to empty hovering spot - 3

Takeoff
2.1 stay still - 0
2.2 takeoff - 1
2.3 move to battery port - 2
2.4 move from battery port - 3


"""


class RandomWalk_ActionManager:
    """
    This class should decode the actions and send the cooridnate to the Main environment
    """
    def __init__(self, port):
        self.port = port
        self.actions = {0 : "stay still",
                        1 : "takeoff",
                        2 : "normal-1",
                        3 : "normal-2",
                        4 : "battery-1",
                        5 : "hover-1",
                        6 : "hover-2",
                        7 : "hover-3",
                        8 : "hover-4",
                        9 : "continue",
                        10 : "avoid collision"}

    
    def action_decode(self,drone, action):
        """
        action count:

            staystill - 0
            takeoff - 1
            move to normal port 1 - 2
            move to normal port 2 - 3
            move to battery port 1 - 4
            move to hover spot 1 - 5
            move to hover spot 2 - 6
            move to hover spot 3 - 7
            move to hover spot 4 - 8
            continue moving - 9
            avoid collision - 10
            

            
    
        First check the status then assign the port/hoveringspot/takeoff/land .... etc
        Update: We will use masking so no need to check the status
        Updateupdate: if using random walk might need to add extra conditionals to the if-elif statements

        Parameters
        ----------
        drone : drone object
            DESCRIPTION.
        action : just a scalar value [0-4]
            DESCRIPTION.


        Returns
        -------
        None.

        """
        #action = self.actions[action]   # Converting to a string for readability. 
        status = drone.status
        action_possible = True
        

        if status == drone.all_states['in-air']:
            if action == "stay still":
                return {"action": "stay", "position" : None}
            
            elif action ==2:            #need to check if port is occupied before doing this action #todo
                #move/land in normal port - 1
                if self.port.get_port_status(0, 'normal') is False:                    
                    
                    final_pos = self.get_final_pos(self.port.port_status[0]["position"], drone.offset)
                    drone.in_battery_port = 0
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'normal','port_no':0}
                    #self.port.update_port(drone.port_identification)
                    self.port.change_status_normal_port(0, True)
                    return {"position" : final_pos, "action": "land"}
                else:
                    action_possible = False
            elif action ==3:            #need to check if port is occupied before doing this action #todo
                #move/land in normal port - 1
                if self.port.get_port_status(1, 'normal') is False:   
                    
                    final_pos = self.get_final_pos(self.port.port_status[1]["position"], drone.offset)
                    drone.in_battery_port = 0
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'normal','port_no':1}
                    #self.port.update_port(drone.port_identification)
                    self.port.change_status_normal_port(1, True)
                    return {"position" : final_pos, "action": "land"}
                else:
                    action_possible = False
            
            elif action ==4:            #need to check if port is occupied before doing this action #todo
                #move/land in normal port - 1
                if self.port.get_port_status(0, 'battery') is False:   
                    
                    final_pos = self.get_final_pos(self.port.battery_port_status[0]["position"], drone.offset)
                    drone.in_battery_port = 1
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'battery', 'port_no':0}
                    self.port.change_status_battery_port(0, True)
                    #self.port.update_port(drone.port_identification)
                    return {"position" : final_pos, "action": "land-b"}        
                else:
                    action_possible = False
                    
            elif action in [5, 6, 7, 8]: #Hovering actions
                hport = action - 5
                if not self.port.get_port_status(hport, "hover"):
                    final_pos = self.get_final_pos(self.port.hover_spot_status[hport]["position"], drone.offset)
                    drone.in_battery_port = 0
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'hover','port_no':hport}
                    self.port.change_hover_spot_status(0, True)
                    return {"position" : final_pos, "action": "hover"}
                else:
                    action_possible = False
            else:
                #print(f"action {action} not available")
                return {"action": "stay", "position" : None}
        elif status == drone.all_states['in-port'] or status == drone.all_states['battery-port']:
            if action == 0:
                return {"action": "stay", "position" : None}


            elif action == 1:
                #takeoff
                #dest = self.port.get_destination(choice=0)
                dest = drone.upcoming_schedule["end-port"]
                final_pos = self.get_final_pos(dest, drone.offset)
                self.port.update_port(drone.port_identification)
                return {"position" : final_pos, "action": "takeoff"}
            
            
            elif action ==2:            #need to check if port is occupied before doing this action #todo
                if self.port.get_port_status(0, 'normal') is False: 
                    final_pos = self.get_final_pos(self.port.port_status[0]["position"], drone.offset)
                    drone.in_battery_port = 0
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'normal','port_no':0}
                    self.port.change_status_normal_port(0, True)
                    return {"position" : final_pos, "action": "move"}
            
            elif action ==3:            #need to check if port is occupied before doing this action #todo
                if self.port.get_port_status(1, 'normal') is False: 
                    
                    final_pos = self.get_final_pos(self.port.port_status[1]["position"], drone.offset)
                    drone.in_battery_port = 0
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'normal','port_no':1}
                    self.port.change_status_normal_port(1, True)
                    return {"position" : final_pos, "action": "move"}
                else:
                    action_possible = False
            elif action ==4:            #need to check if port is occupied before doing this action #todo
                if self.port.get_port_status(0, 'battery') is False: 
                    final_pos = self.get_final_pos(self.port.battery_port_status[0]["position"], drone.offset)
                    drone.in_battery_port = 1
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'battery', 'port_no':0}
                    self.port.change_status_battery_port(0, True)
                    return {"position" : final_pos, "action": "move-b"}      
                else:
                    action_possible = False
            ###########################################################################
        else:
                return {"action": "stay", "position" : None}
        if action_possible == False:
            return {'action':'continue'}
        return {'action':'continue'} # In the case that step is called while a robot is in-action
    
    def get_final_pos(self,port, offset):
        return [port[0] + offset[0] , port[1] + offset[1], port[2]]
