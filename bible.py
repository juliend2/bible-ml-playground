
def get_scripture_id(book, chapter, verse):
    book = book.lower().replace(' ', '')
    return '%s-%d-%d' % (book, chapter, verse)
