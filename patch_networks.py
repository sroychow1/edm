import re

with open("training/networks.py", "r") as f:
    text = f.read()

# 1. Duplicate Conv2d -> RowConv2d
conv2d_code = re.search(r"@persistence\.persistent_class\nclass Conv2d.*?return x", text, re.DOTALL).group(0)

row_conv2d_code = conv2d_code.replace("class Conv2d", "class RowConv2d")
row_conv2d_code = row_conv2d_code.replace(
    "out_channels, in_channels, kernel, kernel", "out_channels, in_channels, 1, kernel"
)
row_conv2d_code = row_conv2d_code.replace(
    "fan_in=in_channels*kernel*kernel", "fan_in=in_channels*1*kernel"
)
row_conv2d_code = row_conv2d_code.replace(
    "fan_out=out_channels*kernel*kernel", "fan_out=out_channels*1*kernel"
)

# Resample filter
row_conv2d_code = row_conv2d_code.replace(
    "f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()",
    "f.unsqueeze(0).unsqueeze(1).unsqueeze(2) / f.sum()"
)

# Padding & Stride in forward
row_conv2d_code = row_conv2d_code.replace(
    "padding=max(f_pad - w_pad, 0)", "padding=(0, max(f_pad - w_pad, 0))"
)
row_conv2d_code = row_conv2d_code.replace(
    "padding=max(w_pad - f_pad, 0)", "padding=(0, max(w_pad - f_pad, 0))"
)
row_conv2d_code = row_conv2d_code.replace(
    "padding=w_pad+f_pad", "padding=(0, w_pad+f_pad)"
)
row_conv2d_code = row_conv2d_code.replace(
    "padding=f_pad", "padding=(0, f_pad)"
)
row_conv2d_code = row_conv2d_code.replace(
    "padding=w_pad", "padding=(0, w_pad)"
)

row_conv2d_code = row_conv2d_code.replace("stride=2", "stride=(1, 2)")
# It appears twice for up and down... actually let's just replace all stride=2
row_conv2d_code = row_conv2d_code.replace("stride=2", "stride=(1, 2)") # already done by above
row_conv2d_code = row_conv2d_code.replace("f.mul(4)", "f.mul(2)")


# 2. Duplicate UNetBlock -> RowUNetBlock
unetblock_code = re.search(r"@persistence\.persistent_class\nclass UNetBlock.*?return x", text, re.DOTALL).group(0)
row_unetblock_code = unetblock_code.replace("class UNetBlock", "class RowUNetBlock")
row_unetblock_code = row_unetblock_code.replace("Conv2d(", "RowConv2d(")

# 3. Duplicate SongUNet -> RowSongUNet
songunet_code = re.search(r"@persistence\.persistent_class\nclass SongUNet.*?return aux", text, re.DOTALL).group(0)
row_songunet_code = songunet_code.replace("class SongUNet", "class RowSongUNet")
row_songunet_code = row_songunet_code.replace("Conv2d(", "RowConv2d(")
row_songunet_code = row_songunet_code.replace("UNetBlock(", "RowUNetBlock(")
row_songunet_code = row_songunet_code.replace("isinstance(block, UNetBlock)", "isinstance(block, RowUNetBlock)")


text += "\n\n" + "#" + "-"*76 + "\n# Modified 1D Row versions of convolution and UNet layers\n\n"
text += row_conv2d_code + "\n\n" + row_unetblock_code + "\n\n" + row_songunet_code + "\n"

with open("training/networks.py", "w") as f:
    f.write(text)

