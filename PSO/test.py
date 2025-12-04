
from roboticstoolbox.robot import Chain
from roboticstoolbox.robot import Link

link1 = Link(d=0.1, a=0, alpha=0)
link2 = Link(d=0, a=0.5, alpha=1.57)
chain = Chain([link1, link2], name="test_arm")

print("Chain创建成功，关节数：", chain.n)
print("Chain类导入正常！")