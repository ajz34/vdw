from pyscf import lib


def wrapper(vdw_cls, mf, return_instance=True, **kwargs):
    with_vdw = vdw_cls(mf, **kwargs)
    mf_cls = mf.__class__

    class VDWInner(vdw_cls, mf_cls, lib.StreamObject):
        def __init__(self, mf, with_vdw):
            self.__dict__.update(mf.__dict__)
            self.with_vdw = with_vdw
            self._keys.update("with_vdw")

        def dump_flags(self, verbose=0):
            mf_cls.dump_flags(self, verbose)
            if self.with_vdw:
                self.with_vdw.dump_flags(verbose)
            return self

        def energy_nuc(self):
            enuc = mf_cls.energy_nuc(self)
            evdw = self.with_vdw.eng
            self.e_vdw = evdw
            if self.with_vdw:
                enuc += evdw
            return enuc

        def reset(self, mol=None):
            if mol is not None:
                self.with_vdw.mol = mol
            return mf_cls.reset(self, mol)

        def nuc_grad_method(self):
            scf_grad = mf_cls.nuc_grad_method(self)
            return grad_vdw(scf_grad)

        Gradients = lib.alias(nuc_grad_method, alias_name='Gradients')

    if return_instance:
        return VDWInner(mf, with_vdw)
    else:
        return VDWInner


def grad_vdw(mf_grad):
    from pyscf.grad import rhf as rhf_grad
    assert(isinstance(mf_grad, rhf_grad.Gradients))
    # ensure that the zeroth order results include vdw corrections
    if not getattr(mf_grad.base, 'with_vdw', None):
        raise ValueError("This object is not a vdw correction!")

    cls_mf_grad = mf_grad.__class__
    with_vdw = mf_grad.base.with_vdw

    class VDWGradInner(with_vdw.__class__, cls_mf_grad):
        def grad_nuc(self, mol=None, atmlst=None):
            nuc_g = cls_mf_grad.grad_nuc(self, mol, atmlst)
            vdw_g = with_vdw.grad
            if atmlst is not None:
                vdw_g = vdw_g[atmlst]
            nuc_g += vdw_g
            return nuc_g

    vdw_grad = VDWGradInner.__new__(VDWGradInner)
    vdw_grad.__dict__.update(mf_grad.__dict__)
    return vdw_grad
