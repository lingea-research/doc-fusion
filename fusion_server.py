
from sentence_transformers import SentenceTransformer
import fusion

import sys
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi

ENCODER = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

def display(collection):
    rows = collection
    result = ""
    for row_id, row in enumerate(rows):
        result += "TOPIC: " +  str(row_id + 1) + "\n"
        result += "\n".join(row)
        result += "\n\n"
    return result
    
class RequestHandler(BaseHTTPRequestHandler):
    def _send_response(self, message):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(bytes(message, "utf8"))

    def do_POST(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST',
                     'CONTENT_TYPE':self.headers['Content-Type'],                     
                     })

        plaintext = form.getfirst("data")
        n_clusters = int(form.getfirst("n_clusters"))
        n_tokens = int(form.getfirst("n_tokens"))
        full_textrank = bool(form.getfirst("full_textrank", default=False))
        method = form.getfirst("method", default="agglomerativeClustering")
        seeds = form.getfirst("seeds", default="").split("\n")
        
        print("Plaintext:\n", plaintext)
        print("N clusters =", n_clusters)
        print("N tokens =", n_tokens)        
        print("full textrans =", full_textrank)
        print("method =", method)
        print("seeds =", seeds)

        clusters = fusion.summarize(plaintext, 
                                    ENCODER, 
                                    full_textrank=full_textrank, 
                                    method=method, 
                                    seeds=seeds, 
                                    n_clusters=n_clusters, 
                                    n_tokens=n_tokens)

        result = ""
        for i in range(len(clusters)):
            cluster = clusters[i]
            result += "TOPIC: " +  str(i + 1) + "\n"
            result += "\n".join(cluster[:n_tokens])
            result += "\n\n"

        self._send_response(result)

def main(server_class=HTTPServer, handler_class=RequestHandler, port=8003):
    
    print("Starting server on port:", port)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print("Server is accepting requests.")
    httpd.serve_forever()
                    
if __name__ == "__main__":
    main()
