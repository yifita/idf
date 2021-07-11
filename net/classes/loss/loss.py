import numpy as np
from config import ConfigObject
import bisect
from numbers import Number

class Loss(ConfigObject):
    def __init__(self, config):
        self.linear_growth : bool = True
        super().__init__(config)

    def _get_loss_weight(self, name, progress):
        weight = getattr(self, name)
        if isinstance(weight, Number):
            return weight

        if not hasattr(self, '_phase_progress'):
            self._phase_progress = {}
            self._phase_value = {}

        if name not in self._phase_progress:
            self._phase_progress[name] = [values[0] for values in weight]
            self._phase_value[name] = [values[1] for values in weight]

        _phase = bisect.bisect_left(self._phase_progress[name], progress)
        if _phase >= len(self._phase_progress[name]):
            return self._phase_value[name][_phase-1]

        if self.linear_growth and _phase > 0:
            # cosine anealing
            v0 = self._phase_value[name][_phase-1]
            p1p0 = (self._phase_progress[name][_phase] - self._phase_progress[name][_phase-1])
            pp0 = progress-self._phase_progress[name][_phase-1]
            assert(pp0/p1p0 <= 1.0 and pp0/p1p0 >= 0.0)
            v1v0 = (self._phase_value[name][_phase] - self._phase_value[name][_phase-1])
            return v0 + (1-np.cos(np.pi*pp0/p1p0)) * v1v0 / 2

        return self._phase_value[name][_phase]

    def __call__(self, model_output : dict):
        pass

