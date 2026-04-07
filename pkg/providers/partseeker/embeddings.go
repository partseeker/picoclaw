// PicoClaw - Ultra-lightweight personal AI agent
// License: MIT
//
// Copyright (c) 2026 PicoClaw contributors

// Package partseeker provides utilities for Partseeker-specific local inference services.
//
// Chat completions are handled via the standard "local/" provider prefix in the factory
// (routes to Gemma 4 at http://clawasaki:8080/v1). This package provides the
// EmbeddingsClient for Qwen text embeddings (http://clawasaki:8081/v1/embeddings),
// which sit outside the chat provider interface.
package partseeker

import (
	"context"
	"time"

	"github.com/partseeker/unified-partseeker-llm/pkg/embeddings"
)

const (
	DefaultEmbeddingsEndpoint = "http://clawasaki:8081/v1/embeddings"
	DefaultEmbeddingsModel    = "Qwen/Qwen3-Embedding-0.6B"
)

// EmbeddingsClient wraps the unified-partseeker-llm embeddings client
// with picoclaw-appropriate defaults (clawasaki endpoint).
type EmbeddingsClient struct {
	inner *embeddings.Client
}

// EmbeddingsOption configures an EmbeddingsClient.
type EmbeddingsOption func(*embeddingsConfig)

type embeddingsConfig struct {
	endpoint   string
	model      string
	timeout    time.Duration
	maxRetries int
}

func WithEmbeddingsEndpoint(endpoint string) EmbeddingsOption {
	return func(c *embeddingsConfig) { c.endpoint = endpoint }
}

func WithEmbeddingsModel(model string) EmbeddingsOption {
	return func(c *embeddingsConfig) { c.model = model }
}

func WithEmbeddingsTimeout(d time.Duration) EmbeddingsOption {
	return func(c *embeddingsConfig) { c.timeout = d }
}

func WithEmbeddingsMaxRetries(n int) EmbeddingsOption {
	return func(c *embeddingsConfig) { c.maxRetries = n }
}

// NewEmbeddingsClient creates an EmbeddingsClient pointing at the Qwen endpoint
// on clawasaki. Override with WithEmbeddingsEndpoint for testing.
func NewEmbeddingsClient(opts ...EmbeddingsOption) *EmbeddingsClient {
	cfg := &embeddingsConfig{
		endpoint:   DefaultEmbeddingsEndpoint,
		model:      DefaultEmbeddingsModel,
		maxRetries: 3,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	innerOpts := []embeddings.ClientOption{
		embeddings.WithEndpoint(cfg.endpoint),
		embeddings.WithModel(cfg.model),
		embeddings.WithMaxRetries(cfg.maxRetries),
	}
	if cfg.timeout > 0 {
		innerOpts = append(innerOpts, embeddings.WithTimeout(cfg.timeout))
	}

	return &EmbeddingsClient{inner: embeddings.New(innerOpts...)}
}

// Embed returns embedding vectors for the given texts, in input order.
// Returns an empty slice without making an HTTP call when texts is empty.
func (c *EmbeddingsClient) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	return c.inner.Embed(ctx, texts)
}
