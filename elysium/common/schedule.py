from typing import Union, Dict, List, Tuple

class Schedule: 
    def __init__(self, schedule: Union[Dict, int]) -> None:
        """A schedule takes progess as an input (eg. steps, episodes, generations) and returns a value for a hyperparameter (e.g lr)
        Args:
            schedule (Dict): often defined in confing files:
            in yaml the dict should have the following structure:
            {
                schedule_type: 'steps', (or generations, reward etc)
                interpolation: 'linear', (or none -> results in step function)
                schedule: 
                    [0, 0.5],
                    [1e3, 0.4],
                    [2e3, 0.3],
                    [3e3, 0.2],
                    [1e4, 0.1]
                    
            }
        """
        
        # if the schedule is just a number entered -> only return that value 
        self.is_static = isinstance(schedule, int)
        
        if self.is_static:
            self.static_value = schedule
        else:
            self.schedule_type: str = schedule['schedule_type'] 
            self.schedule: Union[List[List[float]], List[Tuple[int]]] = schedule['schedule']
        
            #schedules with no values will be rejected
            assert len(self.schedule) != 0, "The schedule must have at least one value"
        
        
    def __call__(self, query: Union[float, int]):
        """Get a value for a query value -> This should not be executed too often, because it is not trivial to compute
        Args:
            query (Union[float, int]): [description]
        """
        
        if self.is_static:
            return self.static_value
        
        # get the lowest query value 
        if query > self.schedule[-1][0]:
            # query value is below smalles schedule value -> no extrapolation
            return self.schedule[-1][1] 
        
        elif query < self.schedule[0][0]:
            
            # query value is above biggest schedule value -> no extrapolation
            return self.schedule[0][1]
        else:  
            
            lower = 0
            upper = 0
            
            for i in range(len(self.schedule) - 1):
                if self.schedule[i][0] <= query and self.schedule[i+1][0] >= query:
                    return self.schedule[i][1]
                    # Todo: add interpolation