use crate::Value64;
use anyhow::Result;
use std::any::Any;
use std::collections::HashMap;

pub type NativeFn = Box<
    dyn Fn(&[Value64], &mut HandleRegistry) -> Result<Value64> + Send + Sync,
>;

pub struct NativeFunction {
    pub name: String,
    pub arity: usize,
    pub function: NativeFn,
}

pub struct HandleRegistry {
    handles: Vec<Option<Box<dyn Any + Send + Sync>>>,
    free_list: Vec<u32>,
}

impl HandleRegistry {
    pub fn new() -> Self {
        Self {
            handles: Vec::new(),
            free_list: Vec::new(),
        }
    }

    pub fn store<T: Any + Send + Sync + 'static>(&mut self, value: T) -> u32 {
        if let Some(index) = self.free_list.pop() {
            self.handles[index as usize] = Some(Box::new(value));
            index
        } else {
            let index = self.handles.len() as u32;
            self.handles.push(Some(Box::new(value)));
            index
        }
    }

    pub fn get<T: Any + Send + Sync + 'static>(
        &self,
        index: u32,
    ) -> Option<&T> {
        self.handles
            .get(index as usize)
            .and_then(|opt| opt.as_ref())
            .and_then(|boxed| boxed.downcast_ref::<T>())
    }

    pub fn get_mut<T: Any + Send + Sync + 'static>(
        &mut self,
        index: u32,
    ) -> Option<&mut T> {
        self.handles
            .get_mut(index as usize)
            .and_then(|opt| opt.as_mut())
            .and_then(|boxed| boxed.downcast_mut::<T>())
    }

    pub fn free(&mut self, index: u32) {
        if (index as usize) < self.handles.len() {
            self.handles[index as usize] = None;
            self.free_list.push(index);
        }
    }
}

impl Default for HandleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub struct NativeRegistry {
    pub functions: HashMap<String, NativeFunction>,
    pub handles: HandleRegistry,
}

impl NativeRegistry {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            handles: HandleRegistry::new(),
        }
    }

    pub fn register<F>(&mut self, name: &str, arity: usize, function: F)
    where
        F: Fn(&[Value64], &mut HandleRegistry) -> Result<Value64>
            + Send
            + Sync
            + 'static,
    {
        self.functions.insert(
            name.to_string(),
            NativeFunction {
                name: name.to_string(),
                arity,
                function: Box::new(function),
            },
        );
    }

    pub fn get_function(&self, name: &str) -> Option<&NativeFunction> {
        self.functions.get(name)
    }

    pub fn function_names(&self) -> Vec<String> {
        self.functions.keys().cloned().collect()
    }
}

impl Default for NativeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_registry_store_and_get() {
        let mut registry = HandleRegistry::new();
        let handle = registry.store(42i64);
        assert_eq!(registry.get::<i64>(handle), Some(&42));
    }

    #[test]
    fn test_handle_registry_free_and_reuse() {
        let mut registry = HandleRegistry::new();
        let handle1 = registry.store(1i64);
        let handle2 = registry.store(2i64);
        registry.free(handle1);
        let handle3 = registry.store(3i64);
        assert_eq!(handle3, handle1);
        assert_eq!(registry.get::<i64>(handle3), Some(&3));
        assert_eq!(registry.get::<i64>(handle2), Some(&2));
    }

    #[test]
    fn test_native_registry_register_and_call() {
        let mut registry = NativeRegistry::new();
        registry.register("add", 2, |args, _handles| {
            let left = args[0].as_i64();
            let right = args[1].as_i64();
            Ok(Value64::Integer(left + right))
        });

        assert_eq!(registry.get_function("add").unwrap().arity, 2);
        assert!(registry.functions.contains_key("add"));
    }

    #[test]
    fn test_handle_with_custom_type() {
        struct Window {
            width: u32,
            height: u32,
        }

        let mut registry = HandleRegistry::new();
        let handle = registry.store(Window {
            width: 800,
            height: 600,
        });

        let window = registry.get::<Window>(handle).unwrap();
        assert_eq!(window.width, 800);
        assert_eq!(window.height, 600);
    }
}
