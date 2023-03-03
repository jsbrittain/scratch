export function sampleMiddleware({ dispatch }) {
  return function(next) {
	return function(action) {
	  // Business logic
	  return next(action)
	}
  }
}
