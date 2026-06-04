from dataclasses import dataclass

@dataclass
class Stages:
    stage: str = 'phase1'

    def increment(self):
        transitions = {
            'phase1': 'phase2',
            'phase2': 'halt'
        }
        # Updates the stage if a transition exists; otherwise, keeps the current stage ('halt')
        self.stage = transitions.get(self.stage, self.stage)