from fastai.learner import Metric
import torch
import numpy as np
from fastai.torch_core import flatten_check

class Dice_th_pred(Metric):
    """
    Класс для расчета коэффициента Dice на наборе данных с использованием различных пороговых значений.

    Args:
        ths (numpy.ndarray, optional): Массив значений порога для расчета коэффициента Dice. По умолчанию np.arange(0.1,0.9,0.01).
        axis (int, optional): Ось, по которой будет рассчитываться коэффициент Dice. По умолчанию 1.

    Attributes:
        axis (int): Ось, по которой будет рассчитываться коэффициент Dice.
        ths (numpy.ndarray): Массив значений порога для расчета коэффициента Dice.
        inter (torch.Tensor): Тензор для хранения значений числителя коэффициента Dice.
        union (torch.Tensor): Тензор для хранения значений знаменателя коэффициента Dice.

    Methods:
        reset(): Сбрасывает значения тензоров inter и union до нулевых.
        accumulate(p, t): Рассчитывает значения числителя и знаменателя коэффициента Dice на пакете данных.
        value: Возвращает коэффициент Dice для каждого порога.
    """
    def __init__(self, ths=np.arange(0.1,0.9,0.01), axis=1): 
        self.axis = axis
        self.ths = ths
        self.reset()
        
    def reset(self): 
        self.inter = torch.zeros(len(self.ths))
        self.union = torch.zeros(len(self.ths))
        
    def accumulate(self,p,t):
        """
        Рассчитывает значения числителя и знаменателя коэффициента Dice на пакете данных.

        Args:
            p (torch.Tensor): Тензор прогнозов модели.
            t (torch.Tensor): Тензор истинных меток.

        Returns:
            None.
        """
        pred, targ = flatten_check(p, t)
        for i, th in enumerate(self.ths):
            p = (pred > th).float() 
            self.inter[i] += (p*targ).float().sum().item()
            self.union[i] += (p+targ).float().sum().item()

    @property
    def value(self):
        """
        Возвращает коэффициент Dice для каждого порога.

        Returns:
            torch.Tensor: Тензор с коэффициентами Dice для каждого порога.
        """
        dices = torch.where(self.union > 0.0, 2.0*self.inter/self.union, 
                            torch.zeros_like(self.union))
        return dices

class Dice_soft(Metric):
    '''
    Класс для вычисления метрики Dice на мягких (вероятностных) предсказаниях модели.

    Аргументы:

    axis (int, optional): Ось для вычисления метрики. По умолчанию 1.
    Атрибуты:

    axis (int): Ось для вычисления метрики.
    Методы:

    reset(): Сброс значений метрики перед вычислением на новых данных.
    accumulate(learn): Вычисление метрики на одном батче данных.
    learn: Обучающая модель.
    value: Вычисление значения метрики на всех данных. Возвращает значение метрики в виде скаляра, либо None, если знаменатель равен 0.
    '''
    def __init__(self, axis=1): 
        self.axis = axis 
    def reset(self): self.inter,self.union = 0,0
    def accumulate(self, learn):
        pred,targ = flatten_check(torch.sigmoid(learn.pred), learn.y)
        self.inter += (pred*targ).float().sum().item()
        self.union += (pred+targ).float().sum().item()
    @property
    def value(self): return 2.0 * self.inter/self.union if self.union > 0 else None
    
# dice with automatic threshold selection
class Dice_th(Metric):
    '''
    Класс Dice_th вычисляет значение метрики Dice для различных пороговых значений для бинаризации прогнозов.

    Аргументы:

    ths (np.ndarray): массив значений пороговых значений, используемых для бинаризации предсказаний (по умолчанию np.arange(0.1,0.9,0.05))
    axis (int): ось, по которой происходит вычисление (по умолчанию 1)
    Атрибуты:

    axis (int): ось, по которой происходит вычисление
    ths (np.ndarray): массив значений пороговых значений
    Методы:

    reset(): обнуляет значения для накопления метрики для следующего вычисления
    accumulate(learn): вычисляет значение метрики Dice для данной пары предсказаний и целевых значений, используя пороговые значения
    value(): возвращает максимальное значение метрики Dice из значений, вычисленных для каждого порога, если хотя бы для одного порога значение было вычислено, иначе None.
    '''
    def __init__(self, ths=np.arange(0.1,0.9,0.05), axis=1): 
        self.axis = axis
        self.ths = ths
        
    def reset(self): 
        self.inter = torch.zeros(len(self.ths))
        self.union = torch.zeros(len(self.ths))
        
    def accumulate(self, learn):
        pred,targ = flatten_check(torch.sigmoid(learn.pred), learn.y)
        for i,th in enumerate(self.ths):
            p = (pred > th).float()
            self.inter[i] += (p*targ).float().sum().item()
            self.union[i] += (p+targ).float().sum().item()

    @property
    def value(self):
        dices = torch.where(self.union > 0.0, 
                2.0*self.inter/self.union, torch.zeros_like(self.union))
        return dices.max()

