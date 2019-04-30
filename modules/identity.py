import torch

class Identity(torch.jit.ScriptModule):
    r"""A placeholder identity operator that is argument-insensitive.
     Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)
     Examples::
         >>> m = nn.Identity(54, unused_argumenbt1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])
     """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    @torch.jit.script_method
    def forward(self, input):
        return input