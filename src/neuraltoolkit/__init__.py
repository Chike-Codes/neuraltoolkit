# Copyright (C) 2026  <Your Name or Organization>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://gnu.org>.

__version__ = "1.0.1"

from .modules import *

from .data import *
from .core import *
from .modules.layers import *
from .loss import *
from .optimizers import *
from .initializers import *
from .training import Trainer
from . import datasets


print("Neural Tool Kit loaded!")