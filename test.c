int a[256], b[256], c[256];

void vectorized () {
  for (int i=0; i<256; i++){
    a[i] = b[i] + c[i];
  }
}

void not_vectorized () {
  for (int i=0; i<256; i++){
    a[i] = i%2==1 ? b[i] + c[i] : 0;
  }
}

int main(){
    // vectorized();
    not_vectorized();
}


