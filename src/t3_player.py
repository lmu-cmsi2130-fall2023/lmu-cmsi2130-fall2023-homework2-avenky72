"""
Artificial Intelligence responsible for playing the game of T3!
Implements the alpha-beta-pruning mini-max search algorithm
Atul Venkatesan
"""
from dataclasses import *
from typing import *
from t3_state import *

@dataclass
class Move:
    ut_score: float
    action: Optional["T3Action"] 
    depth: float

    """The method that gets two moves and decides which is the 'better' one to return
    """
def tiebreak(self: "Move", other_score: float, other_depth: float, other_act: Optional["T3Action"] , is_max: bool) -> bool:
    if not self.ut_score == other_score: 
        return self.ut_score < other_score if is_max else self.ut_score > other_score
    if not self.depth == other_depth: 
        return self.depth > other_depth
    return self.action > other_act if not (self.action is None or other_act is None) else False

    """
    returns the action
    """
def choose(state: "T3State") -> Optional["T3Action"]:
    alphabeta = minimax(state, float('-inf'), float('inf'), True, 0)
    return alphabeta.action
    
    """
    Minimax is the alpha-beta pruning function.
    It first checks all the terminal states.
    Then during the Max Turn, it loops through all the actions and states from the getTransitions generator
    and recursively checks it against the best case. 
    If it beats the best case, the best case takes it values.
    The min does the opposite, taking the lowest score.
    """
def minimax (State: "T3State", a: float, b: float, max_turn: bool, depth: int) -> Move:
    if State.is_win() and max_turn:
        return Move(0, None, depth)
    elif State.is_win() and not max_turn:
        return Move(2, None, depth)
    if State.is_tie():
        return Move(1, None, depth)
    
    best_action = Move(float('-inf') if max_turn else float('inf'), None, float('-inf')) 
    if max_turn: 
        for action, stat in State.get_transitions():
            children1 = minimax(stat, a, b, (not max_turn), depth+1)
            if tiebreak(best_action, children1.ut_score, children1.depth, action, True):
                best_action = Move(children1.ut_score, action, children1.depth)
            a = max(a, best_action.ut_score)
            if b <= a:
                break
        return best_action
    else:
        for action, stat in State.get_transitions():
            children2 = minimax(stat, a, b, (not max_turn), depth+1)
            if tiebreak(best_action, children2.ut_score, children2.depth, action, False):
                best_action = Move(children2.ut_score, action, children2.depth)
            b = min(b, best_action.ut_score)
            if b <= a:
                break
        return best_action   
