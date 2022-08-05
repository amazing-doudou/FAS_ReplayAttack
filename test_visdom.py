import ssl
import urllib2


context = ssl._create_unverified_context()
print (urllib2.urlopen("https://www.12306.cn/mormhweb/", context=context).read())
