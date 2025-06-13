# test_blackboard.py
import pytest
import threading
from blackboard import Blackboard, EventType
import time

def test_concurrent_updates():
    bb = Blackboard()
    
    def mock_agent(section, data):
        for i in range(100):
            time.sleep(0.1)
            bb.update(section, f"{data}_{i}", "test_agent", False)
    
    threads = [
        threading.Thread(target=mock_agent, args=('section1', 'data1')),
        threading.Thread(target=mock_agent, args=('section2', 'data2'))
    ]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert bb.get('section1') == "data1_99"
    assert bb.get('section2') == "data2_99"
    
    for i in bb.audit_log:
        print(i)

def test_notifications():
    bb = Blackboard()
    events_received = []
    
    def callback():
        events_received.append(1)
    
    bb.subscribe(EventType.REQUIREMENTS_UPDATED, callback)
    bb.update('user_requirements', {'test': 1}, 'test_agent')
    
    time.sleep(0.1)  # Dar tiempo a la notificación asíncrona
    assert len(events_received) == 1
    
    for i in bb.audit_log:
        print(i)
    
test_notifications()