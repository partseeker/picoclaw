// PicoClaw - Ultra-lightweight personal AI agent
// License: MIT
//
// Copyright (c) 2026 PicoClaw contributors

package partseeker_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/sipeed/picoclaw/pkg/providers/partseeker"
)

// embedResponse builds a fake /v1/embeddings response body.
// Results are returned in reversed index order to exercise the sort-by-index path.
func embedResponse(vecs [][]float32) string {
	type item struct {
		Object    string    `json:"object"`
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	}
	data := make([]item, len(vecs))
	for i, v := range vecs {
		data[i] = item{Object: "embedding", Index: i, Embedding: v}
	}
	// Reverse to verify sort-by-index is applied.
	for i, j := 0, len(data)-1; i < j; i, j = i+1, j-1 {
		data[i], data[j] = data[j], data[i]
	}
	b, _ := json.Marshal(map[string]any{"object": "list", "data": data})
	return string(b)
}

func TestEmbeddingsClient_RequestFormat(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"object":"list","data":[]}`))
	}))
	defer srv.Close()

	c := partseeker.NewEmbeddingsClient(partseeker.WithEmbeddingsEndpoint(srv.URL))
	_, _ = c.Embed(context.Background(), []string{"hello", "world"})

	if gotBody["model"] != partseeker.DefaultEmbeddingsModel {
		t.Errorf("model = %v, want %v", gotBody["model"], partseeker.DefaultEmbeddingsModel)
	}
	inputs, _ := gotBody["input"].([]any)
	if len(inputs) != 2 {
		t.Errorf("input len = %d, want 2", len(inputs))
	}
	if inputs[0] != "hello" || inputs[1] != "world" {
		t.Errorf("input = %v, want [hello world]", inputs)
	}
}

func TestEmbeddingsClient_ContentTypeHeader(t *testing.T) {
	var gotContentType string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotContentType = r.Header.Get("Content-Type")
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"object":"list","data":[]}`))
	}))
	defer srv.Close()

	c := partseeker.NewEmbeddingsClient(partseeker.WithEmbeddingsEndpoint(srv.URL))
	_, _ = c.Embed(context.Background(), []string{"hi"})

	if gotContentType != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", gotContentType)
	}
}

func TestEmbeddingsClient_ReturnsVectorsInInputOrder(t *testing.T) {
	vecs := [][]float32{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
		{0.7, 0.8, 0.9},
	}
	body := embedResponse(vecs) // reversed index order

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(body))
	}))
	defer srv.Close()

	c := partseeker.NewEmbeddingsClient(partseeker.WithEmbeddingsEndpoint(srv.URL))
	got, err := c.Embed(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != len(vecs) {
		t.Fatalf("got %d vectors, want %d", len(got), len(vecs))
	}
	for i := range vecs {
		for j := range vecs[i] {
			if got[i][j] != vecs[i][j] {
				t.Errorf("got[%d][%d] = %v, want %v", i, j, got[i][j], vecs[i][j])
			}
		}
	}
}

func TestEmbeddingsClient_EmptyInput(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer srv.Close()

	c := partseeker.NewEmbeddingsClient(partseeker.WithEmbeddingsEndpoint(srv.URL))
	got, err := c.Embed(context.Background(), []string{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("got %d vectors for empty input, want 0", len(got))
	}
	if called {
		t.Error("HTTP call made for empty input, want none")
	}
}

func TestEmbeddingsClient_Timeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(200 * time.Millisecond)
		w.Write([]byte(`{"object":"list","data":[]}`))
	}))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	c := partseeker.NewEmbeddingsClient(
		partseeker.WithEmbeddingsEndpoint(srv.URL),
		partseeker.WithEmbeddingsMaxRetries(0),
	)
	_, err := c.Embed(ctx, []string{"text"})
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
}

func TestEmbeddingsClient_DefaultEndpoint(t *testing.T) {
	got := partseeker.DefaultEmbeddingsEndpoint
	want := "http://clawasaki:8081/v1/embeddings"
	if got != want {
		t.Errorf("DefaultEmbeddingsEndpoint = %q, want %q", got, want)
	}
}
