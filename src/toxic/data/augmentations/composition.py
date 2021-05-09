import random


class Compose:
    def __init__(self, transforms, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, data, **params):
        if random.random() < self.p:
            for transform in self.transforms:
                if random.random() < transform.p:
                    data = transform(data, **params)
        return data


class OneOf:
    def __init__(self, transforms, p=1.0):
        self.transforms = transforms
        self.p = p
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, data, **params):
        if self.transforms_ps and (random.random() < self.p):
            transform = random.choices(
                self.transforms,
                weights=self.transforms_ps,
                k=1
            )[0]
            data = transform(data, **params)
        return data
