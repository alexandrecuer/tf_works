from src.tools import PHPFina
import matplotlib.pylab as plt

start=1577404800
feed=PHPFina(66,10,"../labo/phpfina")
feed.getMetas()
feed.setStart(start)
feed.getDatas(60000)

plt.subplot(111)
plt.plot(feed._datas)
plt.show()
