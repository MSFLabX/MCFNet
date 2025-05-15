import torch
import torch.nn as nn
import numbers
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FDFA(nn.Module):
    def __init__(self, dim, num_heads, ):
        super(FDFA, self).__init__()
        self.num_heads = num_heads

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv1_1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_1_2 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)


        self.conv2_1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv2_1_2 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)




    def forward(self, x1,x2):
        b, c, h, w = x1.shape
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        attn_111 = self.conv1_1_1(x1)
        attn_112 = self.conv1_1_2(x1)

        attn_211 = self.conv2_1_1(x2)
        attn_212 = self.conv2_1_2(x2)


        out1 = attn_111 + attn_112
        out2 = attn_211 + attn_212
        out1 = self.project_out(out1)
        out2 = self.project_out(out2)
      
        k1 = rearrange(x1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(x1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        k2 = rearrange(x2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(x2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        


        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1) + q1
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2) + q2
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out3) + self.project_out(out4) + x1+x2   

        return out



class Cross_atten(nn.Module):
    def __init__(self, dim, num_heads, ):
        super(Cross_atten, self).__init__()
        self.num_heads = num_heads

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv1_1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_2_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv2_1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv2_2_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)



    def forward(self, x1,x2):
        b, c, h, w = x1.shape
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        attn_111 = self.conv1_1_1(x1)
        attn_121 = self.conv1_2_1(x1)

        attn_211 = self.conv2_1_1(x2)
        attn_221 = self.conv2_2_1(x2)


        out1 = attn_111 + attn_121
        out2 = attn_211 + attn_221
        out1 = self.project_out(out1)
        out2 = self.project_out(out2)
        k1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1) + q1
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2) + q2
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out3) + self.project_out(out4) + x1+x2

        return out
