import urllib.request

im_dir = 'images/'
im_url_base = 'https://www.setgame.com/sites/all/modules/setgame_set/assets/images/new/'

for i in range(1, 82):
    filename = str(i) + '.png'
    url = im_url_base + filename
    print('Downloading %s' % url)
    urllib.request.urlretrieve(url, im_dir + filename)