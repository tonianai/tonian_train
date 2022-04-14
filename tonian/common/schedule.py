 
from typing import Callable, Union, Dict, List, Tuple
from numbers import Number

class Schedule: 
    def __init__(self, schedule: Union[Dict, float]) -> None:
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
        self.is_static = not isinstance(schedule, Dict) 
        
        if not self.is_static:
            self.schedule: Union[List[List[float]], List[Tuple[int]]] = schedule['schedule']
            self.schedule = [(float(schedule_item[0]), float(schedule_item[1])) for schedule_item in self.schedule]
        
            #schedules with no values will be rejected
            assert len(self.schedule) != 0, "The schedule must have at least one value"
        elif isinstance(schedule, float):
            self.static_value = schedule
        elif isinstance(schedule, str):
            self.static_value = float(schedule) 
        else:
            raise ValueError(f"The schedule input type is not acceptable, Type: {type(schedule)}")
        
        
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
                    
ScheduleOrValue = Union[Schedule, Callable]

def schedule_or_callable(input: Union[int, float, dict, str]) -> ScheduleOrValue:
    
    if isinstance(input, Dict): 
        return Schedule(input)
    elif isinstance(input, str):
        val = float(input)
        return lambda _ : val
    elif isinstance(input, float) or isinstance(input, int):
        return lambda _ : input
    else:
        raise Exception('any value in the config schould either be a value or a schedule')
    
    