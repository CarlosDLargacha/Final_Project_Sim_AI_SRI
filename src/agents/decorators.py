# agents/decorators.py
from datetime import datetime
from functools import wraps
from typing import Callable, Any
import traceback

def agent_error_handler(func):
    """Decorador que espera recibir self como primer argumento"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            if hasattr(self, 'blackboard'):
                error_entry = {
                    'agent': self.__class__.__name__,
                    'function': func.__name__,
                    'error_type': type(e).__name__,
                    'message': str(e),
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.now().isoformat()
                }
                if not hasattr(self.blackboard.state, 'errors'):
                    self.blackboard.state['errors'] = []
                self.blackboard.state['errors'].append(error_entry)
                print(f"⚠️ Error en {self.__class__.__name__}.{func.__name__}: {str(e)}")
                print(traceback.format_exc())
            return None
    return wrapper

