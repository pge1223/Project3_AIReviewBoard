with open('oss_test.html', encoding='utf-8') as f:
    content = f.read()
idx = content.find('data-project-id="56958"')
print(content[idx-100:idx+2000])
