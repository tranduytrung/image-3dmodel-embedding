class Material:
    def __init__(self, name, ka=(1.0,1.0,1.0), kd=(1.0,1.0,1.0), ks=(1.0,1.0,1.0), ns=16, illum=2):
        self.name = name
        self.ka = ka
        self.kd = kd
        self.illum = illum
        if illum == 1:
            self.ks = (0.0,0.0,0.0)
        else:
            self.ks = ks
            
        self.ns = ns
        
    def __repr__(self):
        return  f'Name {self.name}\n'\
                f'Ka {self.ka}\n'\
                f'Kd {self.kd}\n'\
                f'Ks {self.ks}\n'\
                f'Ns {self.ns}\n'\
                f'illum {self.illum}'