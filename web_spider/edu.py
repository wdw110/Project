# encoding=utf-8
#author:wdw110
#date:2017年2月25日
#教师招聘信息爬取数据

import os
import re
import time
import urllib2
import datetime
import requests
from bs4 import BeautifulSoup

import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.utils import parseaddr, formataddr

urls = ['http://www.hzedu.gov.cn/sites/main/template/list.aspx?Id=56&classid=3','http://www.hzscjy.com/','http://www.bjqjyj.cn/lmnew_list.aspx?flag=2','http://www.hxjy.com/cms/app/info/cat/index.php/20','http://www.xsedu.zj.cn/sites/main/template/list.aspx?id=252','http://www.jgedu.net/col/col1410/index.html','http://www.gsjy.net/sites/xxgk/template/list.aspx?Id=125','http://jyj.hzxh.gov.cn/col/col1217310/index.html','http://www.xiashaedu.com/ineduportal/Components/news/infoListWap.aspx?id=1383','http://www.djdedu.net/index.php/zszp','http://www.yhjy.gov.cn/Class/class_94/index.html','http://www.wenwu8.com/news/news_37.html']

class Edu(object):
	"""docstring for Edu"""
	def __init__(self):
		self.data = []
		self.dateline = ''
		self.getdate()

	def getdate(self):
		now = datetime.datetime.now()
		aDay = datetime.timedelta(days=-3)
		now = now + aDay
		self.dateline = now.strftime('%Y-%m-%d')

	def getcontent(self,url):
		headers = {'User-Agent':"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36"}##浏览器请求头（大部分网站没有这个请求头会报错、请务必加上哦）
		response = requests.get(url, headers=headers)
		find = re.findall('<meta.*?charset=(.*?)>',response.content)
		if len(find):
			encode = re.findall('utf-8|gb2312|gbk|gb10803',find[0])[0]
			if response.encoding == 'ISO-8859-1':
				response.encoding = encode
		con = response.text.encode('utf-8')
		return con

	def all(self, u_arr):
		def hz(url=u_arr[0]):
			base = 'http://www.hzedu.gov.cn'
			content = BeautifulSoup(self.getcontent(url), 'html.parser')
			cons = content.select('ul.shownews tr.shownewstd')
			for con in cons:
				pattern = re.compile('\d{4}-\d{1,2}-\d{1,2}',re.S)
				date = re.findall(pattern, str(con.select('td[style="padding-right:3px;text-align:right;color:#999999;font-size:12px;"]')))[0]
				date = (datetime.datetime.strptime(date,'%Y-%m-%d')).strftime('%Y-%m-%d')
				con_obj = con.select('a')[0].attrs
				url = base + str(con_obj['href'])
				title = con_obj['title'].encode('utf-8')
				if date >= self.dateline and len(re.findall('招聘|诚邀|诚聘',title)):
					if len(re.findall('中小学',title)):
						self.data.append((title,url,date))
					if not len(re.findall('小学|淳安|幼儿|桐庐|富阳|临安|建德',title)):
						self.data.append((title,url,date))


		def sc(url=u_arr[1]):
			year = str(time.localtime().tm_year)
			base = 'http://www.hzscjy.com'
			content = BeautifulSoup(self.getcontent(url), 'html.parser')
			cons = content.select('li')
			for con in cons:
				url  = base + str(con.a['href'])
				title = con.a.string.encode('utf-8')
				date = re.findall('\[(\d{2}-\d{2})\]',str(con))[0]
				date = year + '-' + date
				if date >= self.dateline and len(re.findall('招聘|诚邀|诚聘',title)):
					if len(re.findall('中小学',title)):
						self.data.append((title,url,date))
					if not len(re.findall('小学|淳安|幼儿|桐庐|富阳|临安|建德',title)):
						self.data.append((title,url,date))


		def bj(url=u_arr[2]):
			content = BeautifulSoup(self.getcontent(url), 'lxml')
			cons = content.select('div.listbox ul li')
			for con in cons:
				url = con.select('span.bt a')[0].attrs['href']
				title = con.select('span.bt')[0].string.encode('utf-8')
				date = str(con.select('span.time')[0].string)
				if date >= self.dateline and len(re.findall('招聘|诚邀|诚聘',title)):
					if len(re.findall('中小学',title)):
						self.data.append((title,url,date))
					if not len(re.findall('小学|淳安|幼儿|桐庐|富阳|临安|建德',title)):
						self.data.append((title,url,date))


		def xc(url=u_arr[3]):
			base = 'http://www.hxjy.com'
			content = BeautifulSoup(self.getcontent(url), 'lxml')
			cons = content.select('table.tb_listOutPut > tr')[0:-1]
			for con in cons:
				tmp = con.select('.tb_list_title')[0]
				url = base + tmp.a['href']
				title = tmp.div.string.encode('utf-8')
				date = '20' + con.select('.tb_list_more')[0].string.encode('utf-8')
				if date >= self.dateline and len(re.findall('招聘|诚邀|诚聘',title)):
					if len(re.findall('中小学',title)):
						self.data.append((title,url,date))
					if not len(re.findall('小学|淳安|幼儿|桐庐|富阳|临安|建德',title)):
						self.data.append((title,url,date))


		def xs(url=u_arr[4]):
			base = 'http://www.xsedu.zj.cn'
			content = BeautifulSoup(self.getcontent(url), 'lxml')
			cons = content.select('div.tw_title')
			date_list = content.select('div.tw_data')
			for i in range(len(cons)):
				url = base + cons[i].a.attrs['href']
				title = cons[i].a.string.encode('utf-8')
				date = re.findall('\d{4}-\d{1,2}-\d{1,2}',date_list[i].string.encode('utf-8'))[0]
				date = (datetime.datetime.strptime(date,'%Y-%m-%d')).strftime('%Y-%m-%d')
				if date >= self.dateline and len(re.findall('招聘|诚邀|诚聘',title)):
					if len(re.findall('中小学',title)):
						self.data.append((title,url,date))
					if not len(re.findall('小学|淳安|幼儿|桐庐|富阳|临安|建德',title)):
						self.data.append((title,url,date))


		def jg(url=u_arr[5]):
			base = 'http://www.jgedu.net'
			html = self.getcontent(url)
			pattern = re.compile('<div style=" overflow:hidden; height:44px;.*?>(.*?)</div>',re.S)
			cons = re.findall(pattern,html)
			for con in cons:
				url,title = re.findall('<a style=" font-family.*?href=\'(.*?)\'.*?title=\'(.*?)\'',con)[0]
				date = re.findall('<span style="float:right;.*?>(.*?)</span>',con)[0]
				url = base + url
				if date >= self.dateline and len(re.findall('招聘|诚邀|诚聘',title)):
					if len(re.findall('中小学',title)):
						self.data.append((title,url,date))
					if not len(re.findall('小学|淳安|幼儿|桐庐|富阳|临安|建德',title)):
						self.data.append((title,url,date))


		def gs(url=u_arr[6]):
			base = 'http://www.gsjy.net'
			content = BeautifulSoup(self.getcontent(url), 'html.parser')
			cons = content.select('td[height="25"]')
			date_list = content.select('td[width="100"]')
			for i in range(len(cons)):
				con_obj = cons[i].a.attrs
				url = base + str(con_obj['href'])
				title = con_obj['title'].encode('utf-8')
				date = re.sub('年|月','-',date_list[i].string.strip("[,]").encode('utf-8')).strip('日')
				date = (datetime.datetime.strptime(date,'%Y-%m-%d')).strftime('%Y-%m-%d')
				if date >= self.dateline and len(re.findall('招聘|诚邀|诚聘',title)):
					if len(re.findall('中小学',title)):
						self.data.append((title,url,date))
					if not len(re.findall('小学|淳安|幼儿|桐庐|富阳|临安|建德',title)):
						self.data.append((title,url,date))


		def xh(url=u_arr[7]):
			year = str(time.localtime().tm_year)
			base = 'http://jyj.hzxh.gov.cn'
			html = self.getcontent(url)
			pattern = re.compile('<tr valign="top">(.*?)</tr>',re.S)
			cons = re.findall(pattern,html)
			#content = BeautifulSoup(self.getcontent(url), 'lxml')
			#cons = content.select('div.default_pgContainer tr[valign="top"]')
			for con in cons:
				url = re.findall('<a.*?href=\'(.*?)\'',con)[0]
				if base not in url:
					url = base + url
				title = re.findall('title=\'(.*?)\'',con)[0]
				date = year + '-' + re.findall('\[(\d{2}-\d{2})\]',con)[0]
				if date >= self.dateline and len(re.findall('招聘|诚邀|诚聘',title)):
					if len(re.findall('中小学',title)):
						self.data.append((title,url,date))
					if not len(re.findall('小学|淳安|幼儿|桐庐|富阳|临安|建德',title)):
						self.data.append((title,url,date))


		def xias(url=u_arr[8]):
			year = str(time.localtime().tm_year)
			base = 'http://www.xiashaedu.com/ineduportal/Components/news/'
			content = BeautifulSoup(self.getcontent(url), 'lxml')
			cons = content.select('tr[height="25"]')
			for con in cons:
				url = base + con.a.attrs['href']
				title = con.a.string.encode('utf-8')
				date = year + '-' + con.select('td[width="21%"]')[0].string.encode('utf-8')
				if date >= self.dateline and len(re.findall('招聘|诚邀|诚聘',title)):
					if len(re.findall('中小学',title)):
						self.data.append((title,url,date))
					if not len(re.findall('小学|淳安|幼儿|桐庐|富阳|临安|建德',title)):
						self.data.append((title,url,date))


		def djd(url=u_arr[9]):
			base = 'http://www.djdedu.net'
			content = BeautifulSoup(self.getcontent(url), 'lxml')
			cons = content.select('div.list-box li')
			for con in cons:
				url = base + con.a['href'].encode('utf-8')
				title = con.a['title'].encode('utf-8')
				date = re.findall('<span>.*?(\d{4}-\d{2}-\d{2}).*?</span>',str(con))[0]
				if date >= self.dateline and len(re.findall('招聘|诚邀|诚聘',title)):
					if len(re.findall('中小学',title)):
						self.data.append((title,url,date))
					if not len(re.findall('小学|淳安|幼儿|桐庐|富阳|临安|建德',title)):
						self.data.append((title,url,date))


		def yh(url=u_arr[10]):
			base = 'http://www.yhjy.gov.cn'
			content = BeautifulSoup(self.getcontent(url), 'html.parser')
			cons = content.select('td[align="left"] > a[target="_blank"]')
			date_list = content.select('td[style="display: block; "]')
			for i in range(len(cons)):
				con_obj = cons[i].attrs
				url = base + con_obj['href'].encode('utf-8')
				title = con_obj['title'].encode('utf-8')
				date = re.findall('\d{4}-\d{2}-\d{2}',date_list[i].encode('utf-8'))[0]
				if date >= self.dateline and len(re.findall('招聘|诚邀|诚聘',title)):
					if len(re.findall('中小学',title)):
						self.data.append((title,url,date))
					if not len(re.findall('小学|淳安|幼儿|桐庐|富阳|临安|建德',title)):
						self.data.append((title,url,date))


		def ww(url=u_arr[11]):
			base = 'http://www.wenwu8.com/'
			html = self.getcontent(url)
			content = BeautifulSoup(self.getcontent(url), 'html.parser')
			cons = content.select('ul#listul li')
			for con in cons:
				url = base + con.a['href'].encode('utf-8')
				title = con.a.string.encode('utf-8')
				date = re.findall('\d{4}-\d{2}-\d{2}',str(con))[0]
				if date >= self.dateline and len(re.findall('招聘|诚邀|诚聘',title)):
					if len(re.findall('中小学',title)):
						self.data.append((title,url,date))
					if not len(re.findall('小学|淳安|幼儿|桐庐|富阳|临安|建德',title)):
						self.data.append((title,url,date))


		hz();sc();xc();xs();bj();jg()
		gs();xh();yh();ww()#;djd();xias()
		print '数据抓取完毕，正在发送邮件中...'


class Send(object):
	"""docstring for Send"""
	def __init__(self, msg):
		self.msg = msg
		
	def format_addr(self, s):
		name, addr = parseaddr(s)
		return formataddr(( \
			Header(name, 'utf-8').encode(), \
			addr.encode('utf-8') if isinstance(addr, unicode) else addr))

	def send_email(self):
		# 第三方 SMTP 服务
		mail_host="smtp.163.com"  #设置服务器
		mail_user="hongweiwei923"    #用户名
		mail_pass="www.640"   #口令 


		sender = 'hongweiwei923@163.com'
		receivers = ['hongweiwei923@163.com'] # 接收邮件，可设置为你的QQ邮箱或者其他邮箱

		message = MIMEText(self.msg, 'plain', 'utf-8')
		message['From'] = self.format_addr(u'吴大维 <%s>' % sender)
		message['To'] =  self.format_addr(u'Hww <%s>' % receivers[0])

		subject = '\xf0\x9f\x92\x93\xf0\x9f\x92\x91'+'招聘信息每日推送'+'\xf0\x9f\x92\x95\xf0\x9f\x92\x9c'
		message['Subject'] = Header(subject, 'utf-8')


		try:
			smtpObj = smtplib.SMTP() 
			smtpObj.connect(mail_host, 25)    # 25 为 SMTP 端口号
			smtpObj.login(mail_user,mail_pass) 
			smtpObj.sendmail(sender, receivers, message.as_string())
			print "邮件发送成功"
		except smtplib.SMTPException as e:
			print e
			print "Error: 无法发送邮件"

if __name__ == '__main__':
	a = Edu()
	a.all(urls)
	news = a.data
	news = sorted(news, key=lambda x:x[2], reverse=True)
	if len(news):
		msg = '标题\t网址\t日期\n'
		for infor in news:
			msg += '-'*40+'\n'
			msg += '\t'.join(infor) + '\n'
	else:
		msg = '💖今天没有招聘信息更新，你要努力准备...💕\n\n'
	msg += '\xf0\x9f\x92\x96'*10 + '\nCreated by: wdw110\n' + '\xf0\x9f\x92\x96'*10
	hww = Send(msg)
	hww.send_email()
		